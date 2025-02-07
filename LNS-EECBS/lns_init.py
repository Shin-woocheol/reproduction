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

def log(msg, log_f): #print + write
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()

def main(args, exp_name, log_f):
    M = args.n_agent
    N = args.n_task
    # scenario = load_scenarios(f'323220_1_{M}_{N}/scenario_1.pkl') #* size,size,obs_density,tasklength,agent,task
    scenario_path = f'{args.scenario_fol}/scenario_{args.scenario_num}.pkl'
    scenario = load_scenarios(f'{scenario_path}')
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
    # print(f"0 EECBS_MAKESPAN: {eecbs_ms} SOC(A*): {soc} MS(A*): {ms} Hungarian time: {h_time:.4f}")
    log(f"0 EECBS_MAKESPAN: {eecbs_ms} SOC(A*): {soc} MS(A*): {ms} Hungarian time: {h_time:.4f}", log_f)
    # vis_ta(graph, agent_pos, tasks, 'HA')

    """
    2nd step: Large Neighborhood Search (iteratively) 로 task 다시 assign하면서 그림.
    """
    itr = 0
    task_status = np.zeros(N, dtype=int)  # not complete : 0 , executing : 1, completed : 2
    lns_count = 1
    tot_lns_time = 0

    max_t = args.max_t
    eval_timer = 0
    eval_num = 1
    while True:
        start = time.time()
        tasks = lns_once(tasks, agent_pos, total_tasks, graph, task_status, init_agent_pos, N = 2)
        lns_time = time.time() - start
        tot_lns_time += lns_time
        eval_timer += lns_time

        if eval_timer >= 5:
            soc, ms = cost(init_agent_pos, tasks, graph)
            eecbs_ms = eecbs_cost(agent_pos, tasks, total_tasks, exp_name, grid.shape[0])
            # print(f"{eval_num*5} EECBS_MAKESPAN: {eecbs_ms} SOC(A*): {soc} MS(A*): {ms} TIMECOST: {lns_time:.4f}")
            log(f"{eval_num*5} EECBS_MAKESPAN: {eecbs_ms} SOC(A*): {soc} MS(A*): {ms} TIMECOST: {lns_time:.4f}", log_f)
            eval_timer -= 5
            eval_num += 1

        if tot_lns_time > max_t:
            break
        itr += 1

        # soc, ms = cost(init_agent_pos, tasks, graph)
        # eecbs_ms = eecbs_cost(init_agent_pos, tasks, total_tasks, exp_name, grid.shape[0])
        # print(f"[{itr}_Solution] EECBS_MAKESPAN: {eecbs_ms}, SOC(A*): {soc}, MS(A*): {ms}, TIMECOST: {lns_time:.4f}")

        if args.wandb:
            wandb.log({
                "LNS_iteration": lns_count,
                "EECBS_MAKESPAN": eecbs_ms,
                "A*_Sum of Cost": soc,
                "A*_MAKESPAN": ms,
                "LNS_time": lns_time
            })

        if args.visualize and itr % 5 == 0:
            vis_ta(graph, agent_pos, tasks, itr)

    soc, ms = cost(init_agent_pos, tasks, graph)
    eecbs_ms = eecbs_cost(init_agent_pos, tasks, total_tasks, exp_name, grid.shape[0])
    # print(f"[{itr}_Solution] EECBS_MAKESPAN: {eecbs_ms}, SOC(A*): {soc}, MS(A*): {ms}, TIMECOST: {lns_time:.4f}")
    log(f"[{itr}_Solution] EECBS_MAKESPAN: {eecbs_ms} SOC(A*): {soc} MS(A*): {ms} TIMECOST: {lns_time:.4f}", log_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_t', type = int, default = 30)
    parser.add_argument('--n_agent', type = int, default= 10, help= "num of agents")
    parser.add_argument('--n_task', type = int, default= 20, help= "num of tasks")
    parser.add_argument('--wandb', action='store_true', help="Enable wandb")
    parser.add_argument('--visualize', action='store_true', help="Enable visualize")
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--scenario_fol', type = str, default = 'test_scenario_seed7777')
    parser.add_argument('--scenario_num', type = int, default = 2)
    args = parser.parse_args()
    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.wandb:
        wandb.init(project = 'LNS-EECBS', group = f"LNS-EECBS || a: {args.n_agent}, t: {args.n_task}",name=f"{exp_name}_seed{args.seed}_max_t{args.max_t}", config = vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    # tasks = main(args, exp_name)

    # 로그 파일 경로 생성
    log_dir = f"./LNS_result/{args.scenario_fol}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"scenario_{args.scenario_num}.txt")
    with open(log_path, "w") as log_f:
        main(args, exp_name, log_f)

# python main.py --wandb --seed 7777 --max_t 30
# python main.py --max_t 30