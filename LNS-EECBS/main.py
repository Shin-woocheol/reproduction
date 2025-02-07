import time
import numpy as np
import subprocess
import wandb
import argparse
import random
import os
import pickle

from LNS.hungarian import hungarian
from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from utils.generate_scenarios import load_scenarios
from utils.generate_scenarios import save_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory
from utils.soc_ms import cost, eecbs_cost
from utils.vis_graph import vis_dist, vis_ta
from LNS.lns import lns_once
from datetime import datetime
from copy import deepcopy

solver_path = "EECBS/"

# EECBS cost 돌리기 위해서 save해둠. 
def save_tasks(tasks, exp_name, seed, max_t):
    os.makedirs("saved", exist_ok=True)  # Ensure the directory exists
    file_path = f"saved/{exp_name}_seed{seed}_max_t{max_t}.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(tasks, f)
    print(f"Tasks saved to {file_path}")

def main(args, exp_name):
    M = args.n_agent
    N = args.n_task
    # scenario = load_scenarios(f'323220_1_{M}_{N}/scenario_1.pkl') #* size,size,obs_density,tasklength,agent,task
    
    scenario = load_scenarios(f'{args.scenario_path}')
    grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
    init_agent_pos = deepcopy(agent_pos)
    total_tasks = [[[x, y]] for x, y in total_tasks]
    #* array map, graph, agent pos list[[]], task list[[[]]]
    save_map(grid, exp_name)

    vis_dist(graph, agent_pos, total_tasks) #* fig folder에 scenario 그림 존재.
    """
    1st step: Hungarian Assignment
    #* hungarian algorithm으로 초기 assign. 
    """
    h_time = time.time()
    _, tasks = hungarian(graph, agent_pos, total_tasks) 
    '''
    tasks : {0: [{34: [[2, 3]]}, {1: [[1, 7]]}, {45: [[4, 25]]}], 1: [{43: [[12, 4]]}, {5: [[12, 3]]}], ...
    '''
    h_time = time.time() - h_time
    soc, ms = cost(agent_pos, tasks, graph) #* 여기서 받은 cost는 주변 agent와 연관 없이 A*로만 받은 것.

    eecbs_ms = eecbs_cost(agent_pos, tasks, total_tasks, exp_name, grid.shape[0])
    print(f"[INIT] EECBS_MAKESPAN: {eecbs_ms}, SOC(A*): {soc}, MS(A*): {ms}, Hungarian time: {h_time:.4f}")
    vis_ta(graph, agent_pos, tasks, 'HA')

    """
    2nd step: Large Neighborhood Search (iteratively) 로 task 다시 assign하면서 그림.
    """
    itr = 0
    do_lns = 1
    task_status = np.zeros(N, dtype=int)  # not complete : 0 , executing : 1, completed : 2
    lns_count = 1
    tot_lns_time = 0

    while True:
        max_t = time.time() + args.max_t
        while do_lns:
            if lns_count >= 2: #LNS를 초기에만 실행하고 싶은 경우.
                break
            zero_count = np.sum(task_status == 0)
            if zero_count == 0: # 재할당 할 수 있는 task없음.
                break
            
            start = time.time()
            tasks = lns_once(tasks, agent_pos, total_tasks, graph, task_status, init_agent_pos, N = 2)
            lns_time = time.time() - start
            tot_lns_time += lns_time

            if tot_lns_time > max_t:
                break
            # if itr > max_itr:
            #     break
            itr += 1

            soc, ms = cost(init_agent_pos, tasks, graph)
            eecbs_ms = eecbs_cost(init_agent_pos, tasks, total_tasks, exp_name, grid.shape[0])
            print(f"[{itr}_Solution] EECBS_MAKESPAN: {eecbs_ms}, SOC(A*): {soc}, MS(A*): {ms}, TIMECOST: {lns_time:.4f}")

            if args.wandb:
                wandb.log({
                    "LNS_iteration": lns_count,
                    "EECBS_MAKESPAN": eecbs_ms,
                    "A*_Sum of Cost": soc,
                    "A*_MAKESPAN": ms,
                    "LNS_time": lns_time
                })

            if args.visualize and itr % 50 == 0:
                vis_ta(graph, agent_pos, tasks, itr)

        do_lns = 0 # free agent가 생기는 경우에만 lns다시 수행.

        curr_tasks_pos = [[] for _ in range(M)]  # 각 agent가 이번 EECBS에서 수행할 목적지(1개)
        assigned_task_id = [-1] * M

        for i in range(M):
            assigned_any = False
            for t_dict in tasks[i]:
                t_id, pos = list(t_dict.items())[0]
                if task_status[t_id] in (0,1): # 이번에 수행할 task는 executing or incomplete task
                    if task_status[t_id] == 0: # 처음 할당
                        task_status[t_id] = 1
                    curr_tasks_pos[i] = pos
                    assigned_task_id[i] = t_id
                    assigned_any = True
                    break
            if not assigned_any: # 해당 agent에 할당된 task는 전부 수행함.
                curr_tasks_pos[i] = [agent_pos[i]]
        
        save_scenario(agent_pos, curr_tasks_pos, exp_name, grid.shape[0], grid.shape[1])
        c = [
            solver_path + "eecbs",
            "-m", solver_path + exp_name + ".map",
            "-a", solver_path + exp_name + ".scen",
            "-o", solver_path + exp_name + ".csv",
            "--outputPaths", solver_path + exp_name + "_paths.txt",
            "-k", str(M),
            "-t", "1", 
            "--suboptimality=1.1"
        ]
        subprocess.run(c, capture_output=True)

        agent_traj = read_trajectory(solver_path + exp_name + "_paths.txt")
        T = np.array([len(t) for t in agent_traj])

        if np.any(T == 1): #free agent가 있는 경우.
            do_lns = 1
            lns_count += 1
            next_t = 2 # 바로 다음 step
        else:
            moving_agents_mask = (T > 1)
            next_t = T[moving_agents_mask].min()

        finished_ag = (T == next_t)
        new_agent_pos = deepcopy(agent_pos)
        for i in range(M):
            if T[i] >= 2:
                new_agent_pos[i] = agent_traj[i][next_t - 1]
            else:
                new_agent_pos[i] = agent_pos[i]

        # 이번에 완료되는 task status update
        for i in range(M):
            if finished_ag[i] and assigned_task_id[i] != -1:
                done_tid = assigned_task_id[i]
                task_status[done_tid] = 2

        agent_pos = new_agent_pos

        if all(task_status[t] == 2 for t in range(N)):
            save_tasks(tasks, exp_name, args.seed, args.max_t)
            return tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_t', type = int, default = 10)
    parser.add_argument('--n_agent', type = int, default= 10, help= "num of agents")
    parser.add_argument('--n_task', type = int, default= 20, help= "num of tasks")
    parser.add_argument('--wandb', action='store_true', help="Enable wandb")
    parser.add_argument('--visualize', action='store_true', help="Enable visualize")
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--scenario_path', type = str, default = 'test_scenario_seed7777/scenario_1.pkl')
    args = parser.parse_args()
    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.wandb:
        wandb.init(project = 'LNS-EECBS', group = f"LNS-EECBS || a: {args.n_agent}, t: {args.n_task}",name=f"{exp_name}_seed{args.seed}_max_t{args.max_t}", config = vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    tasks = main(args, exp_name)

# python main.py --wandb --seed 7777 --max_t 100 --visualize