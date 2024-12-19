import subprocess
import random
import torch
import wandb
import numpy as np
import os
import argparse

from collections import deque
from copy import deepcopy
from datetime import datetime
from nn.ag_util import convert_to_bipartite
from nn.agent import Agent
from utils.generate_scenarios import load_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory, compute_astar
from utils.vis_graph import vis_ta
from utils.utils import fix_seed
from tqdm import tqdm

RANDOM = True
solver_path = "EECBS/"
dec_solver_path = "DecAstar/"

def run_episode(agent, M, N, exp_name, T_threshold, train=True, scenario_dir=None, VISUALIZE=False, n_sample=1):
    #* episode 처음 실행시의 init
    episode_traj = []
    task_finished_bef = np.array([False for _ in range(N)]) #* task마다 complete check array인듯.
    scenario = load_scenarios(scenario_dir) 
    grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
    save_map(grid, exp_name) # EECBS
    itr = 0
    episode_timestep = 0
    joint_action_prev = np.array([0] * M) #* 이전 itr에서의 task assignment task idx
    ag_order = np.arange(M) #* agent priority
    continuing_ag = np.array([False for _ in range(M)]) #* rescheduling시 계속 움직일 agent

    shortest_paths = compute_astar(agent_pos, total_tasks, graph) 
    g = convert_to_bipartite(graph, agent_pos, total_tasks, task_finished_bef, shortest_paths)
    
    while True:
        """ 1.super-agent coordinates agent&task pairs """
        # `task_selected` initialized as the `task_finished` to jointly select task at each event
        best_T = None #* 가장 늦게 끝나는 task까지의 step이 가장 작은 것.
        best_curr_tasks_idx = None #* 현재 assign한 task의 idx
        best_agent_traj = None
        best_curr_tasks_pos = None #* 현재 assign한 task의 position
        max_T = 100
        #* 논문에서의 sample(n) 구현
        for _ in range(n_sample): #* n번 assign을 실행하고 trajectory를 구해서, max trjectory 길이가 가장 작은 task assign을 고르는 것.
            if itr > 0:
                random.shuffle(remaining_ag_idx)
                ag_order = np.array(continuing_ag_idx + remaining_ag_idx)

            curr_tasks_pos = [[] for _ in range(M)]  # in order of init agent idx

            joint_action = agent(g, ag_order, continuing_ag, joint_action_prev, train=train)
            #* 결국 agent를 통해서 나오는 output은 각 agent가 선택한 task list. agent를 index로 해서 value가 task index.
            #* 그리고 ag_order로 append되어서 나옴.
            curr_tasks_idx = [0] * M

            for ag_idx, action in zip(ag_order, joint_action):
                #TODO 제대로 assign되는건지 확인 필요.
                if action == -1: #* order가 앞인 모든 agent가 task가져갔으면
                    task_loc = agent_pos[ag_idx].tolist()
                else:
                    task_loc = g.nodes[action + M].data['original_pos'].squeeze().tolist() #* graph에서 agent 다음 task가 나오게 했으니까 task idx 0부터 시작하는 것에서 골랐으니까 그냥 더해도 될듯? -> 확인 필요
                curr_tasks_pos[ag_idx] = [task_loc] #* 수행할 task의 좌표.
                curr_tasks_idx[ag_idx] = action #* 선택한 task index. 0 ~ n_task-1

            """ 2.pass created agent-task pairs to low level solver """
            #* 현재 task assign대로 eecbs를 위한 .scen파일 작성. #*위에서 정한 task대로 low level solver실행.
            save_scenario(agent_pos, curr_tasks_pos, exp_name, grid.shape[0], grid.shape[1])

            # Run solver #* low level solver 작동 step 수 및, path반환.
            c = [solver_path + "eecbs",
                 "-m",
                 solver_path + exp_name + '.map',
                 "-a",
                 solver_path + exp_name + '.scen',
                 "-o",
                 solver_path + exp_name + ".csv",
                 "--outputPaths",
                 solver_path + exp_name + "_paths.txt",
                 "-k", str(M), "-t", "1", "--suboptimality=1.1"]
            subprocess.run(c, capture_output=True)
            
            # Read solver output
            agent_traj = read_trajectory(solver_path + exp_name + "_paths.txt") #* agent pos to task pos까지의 traj
            T = np.array([len(t) for t in agent_traj]) #각 agent 별 trajectory length
            #* sampling을 여러번 해서 best trajectory 얻음.
            if max_T > max(T): #* 가장 늦게 끝나는 task가 가장 작은 것을 best라고 함. 그치 작은 시간 내에 많이 끝나는 것이니까.
                #! ValueError: max() arg is an empty sequence
                best_T = T
                best_curr_tasks_pos = curr_tasks_pos
                best_agent_traj = agent_traj
                best_curr_tasks_idx = curr_tasks_idx # agent idx에는 agent가 선택한 task idx있음.
                max_T = max(T)

        # Mark finished agent, finished task
        next_t = best_T[best_T > 1].min() #* 각 agent의 step 수 중 1 초과인 것만 남기고 그중 min. 아 trajectory가 시작 노드부터 있어서 무조건 1임. 그래서 다음 step에 끝나는 것은 길이가 2
        #* 아래 array는 다음 끝나는 agent flag.
        finished_ag = (best_T == next_t)
        finished_task_idx = np.array(best_curr_tasks_idx)[finished_ag] #* 다음 끝나는 task idx
        task_finished_aft = deepcopy(task_finished_bef)
        task_finished_aft[finished_task_idx] = True #* 끝나는 task표시.
        episode_timestep += next_t - 1 #*원래 기본 trajectory가 1이니까 -1step 후에 끝나는 것.
        # overwrite output
        agent_pos_new = deepcopy(agent_pos) #* 현재 agent position.
        for ag_idx in ag_order:
            if best_T[ag_idx] > 1: #* 이번에 움직이는 agent에 대해 update
                agent_pos_new[ag_idx] = best_agent_traj[ag_idx][next_t - 1] #* 가장빨리 끝나는 다음 step에 각 agent의 위치.

        # Replay memory 에 transition 저장. Agent position 을 graph 의 node 형태로
        terminated = all(task_finished_aft) #* 모든 task가 finished인지 check.

        # TODO: training detail
        if train: #* score를 prob삼아서 sampling을 통해 action을 정한 경우. replay_mem에 저장. training에서는 tr
            # agent.push(g, best_curr_tasks_idx, ag_order, deepcopy(task_finished_bef), next_t, terminated)
            episode_traj.append([g, best_curr_tasks_idx, next_t])
        #* bipartite graph, 현재 agent별 assigned task, 아마 priority, 이전 task_finished정보, 바로 다음 task끝나는 step, 종료 정보 를 buffer에 담음.
        if VISUALIZE:
            # visualize
            vis_ta(graph, agent_pos, best_curr_tasks_pos, str(itr) + "_assigned", total_tasks=total_tasks,
                   task_finished=task_finished_bef)
            vis_ta(graph, agent_pos_new, best_curr_tasks_pos, str(itr) + "_finished", total_tasks=total_tasks,
                   task_finished=task_finished_aft)

        if terminated: #* 모든 task finished
            return episode_timestep, itr, episode_traj

        # agent with small T maintains previous action
        continuing_ag = (0 < best_T - next_t) * (best_T - next_t < T_threshold) 
        #* 가장 빠른 다음 task종료 후에도 trajectory움직여야 하는 것 중, threshold step보다는 적게 남은 것은 계속 수행. 아닌 것은 rescheduling.
        continuing_ag_idx = continuing_ag.nonzero()[0].tolist() #* 계속 움직이는 것 list로 만듬.
        remaining_ag = list(set(range(M)) - set(continuing_ag_idx)) #* rescheduling 해줘야 하는 agent index.

        #* agent order는 random으로 해줌.
        random.shuffle(remaining_ag)
        remaining_ag_idx = remaining_ag
        if type(remaining_ag_idx) == int:
            remaining_ag_idx = [remaining_ag_idx]
            
        # ========================
        ag_order = np.array(continuing_ag_idx + remaining_ag_idx) #* reschedule해야하는 것을 현재 진행중인 것 뒤에 붙임. => 이렇게 해야 continuing은 같은 task 부여받음.
        assert len(set(ag_order)) == M
        joint_action_prev = np.array(best_curr_tasks_idx, dtype=int)

        agent_pos = agent_pos_new #* 가장 빨리 끝나는 다음task episode time에서의 agent들의 위치. #* update된 agent 위치.
        task_finished_bef = task_finished_aft #* 다음 time에 끝나는 것들.
        shortest_paths = compute_astar(agent_pos, total_tasks, graph) #* 각 agent의 현재 위치에서 모든 task로의 A* 재계산.
        g = convert_to_bipartite(graph, agent_pos, total_tasks, task_finished_bef, shortest_paths)
        #* bipartite graph를 만들고, node feature, edge feature도 넣어줌. 논문에서의 그대로
        itr += 1


def main(args, exp_name):
    agent = Agent() #TODO 내부 고쳐야함.
    for _ in range(args.epoch):
        result = {'train_cost':[], 'train_loss':[], "train_n_assign":[], "eval_cost":[], "eval_n_assign":[]}
        #training
        # for i in tqdm(range(args.n_map_train)):
        #     scenario_dir = '323220_1_{}_{}/scenario_{}.pkl'.format(args.n_agent, args.n_task, i + 1)
        #     #TODO 내부 고쳐야함.
        #     cost, n_assign = run_episode(agent, args.n_agent, args.n_task, exp_name, args.task_threshold, train=True, scenario_dir=scenario_dir, VISUALIZE=args.train_visualize) 
        #     if cost is not None:
        #         loss = agent.learn()
        #         result["train_cost"].append(cost)
        #         result["train_n_assign"].append(n_assign)
        #         result["train_loss"].append(loss)

        for _ in range(args.batch_size):
            for i in range(args.n_map_train):
                scenario_dir = '323220_1_{}_{}/scenario_{}.pkl'.format(args.n_agent, args.n_task, i + 1)
                cost, n_assign, episode_traj = run_episode(agent, args.n_agent, args.n_task, exp_name, args.task_threshold, train=True, scenario_dir=scenario_dir, VISUALIZE=args.train_visualize) 
                agent.push(episode_traj)
                result["train_cost"].append(cost)
                result["train_n_assign"].append(n_assign)
                
        loss = agent.learn()
        result["train_loss"].append(loss)


        for i in tqdm(range(args.n_map_eval)):
            scenario_dir = '323220_1_{}_{}/scenario_{}.pkl'.format(args.n_agent, args.n_task, i + 1)
            cost, n_assign, _ = run_episode(agent, args.n_agent, args.n_task, exp_name, args.task_threshold, train=False, scenario_dir=scenario_dir, VISUALIZE=args.eval_visualize)
            if cost is not None:
                result["eval_cost"].append(cost)
                result["eval_n_assign"].append(n_assign)
        
        if args.wandb:
            wandb.log({'train_loss_mean': np.mean(result["train_loss"]), # episodes 진행한 것에 대한 loss.
                'train_cost_mean': np.mean(result["train_cost"]), # episodes 진행한 것에 대한 tot_cost. makespan
                'eval_cost_mean': np.mean(result["eval_cost"]),
                'train_n_assign_mean': np.mean(result["eval_n_assign"]), # 얼마나 scheduling이 일어났는지 terminate까지.
                'eval_n_assign_mean': np.mean(result["eval_n_assign"])})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type = int, default = 1000)
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--n_map_train', type = int, default= 1, help= "num of map in training")
    parser.add_argument('--n_map_eval', type = int, default= 1, help= "num of map in eval")
    parser.add_argument('--n_task_sample', type = int, default= 1, help= "num of task assignment sample")
    parser.add_argument('--n_agent', type = int, default= 10, help= "num of agents")
    parser.add_argument('--n_task', type = int, default= 20, help= "num of tasks")
    parser.add_argument('--task_threshold', type = int, default= 10, help = "task rescheduling threshold")
    parser.add_argument('--wandb', type = bool, default=True)
    parser.add_argument('--eval_visualize', type = bool, default=False)
    parser.add_argument('--train_visualize', type = bool, default=False)
    parser.add_argument('--lr', type = float, default=1e-5)
    parser.add_argument('--batch_size', type = int, default = 1)
    args = parser.parse_args()

    exp_name = datetime.now().strftime("%Y%m%d_%H%M")
    if args.wandb:
        wandb.init(project = 'GFN_MAPF', group = f"a: {args.n_agent}, t: {args.n_task}",name=exp_name, config = vars(args))

    fix_seed(args.seed)
    main(args, exp_name)

#python main.py --wandb True --n_map_train 10 --lr 0.00005