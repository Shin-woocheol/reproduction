import subprocess
import random
import torch
import wandb
import numpy as np
import os

from collections import deque
from copy import deepcopy
from datetime import datetime
from nn.ag_util import convert_dgl
from nn.agent import Agent
from utils.generate_scenarios import load_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory, compute_astar
from utils.vis_graph import vis_ta

RANDOM = True
solver_path = "EECBS/"
dec_solver_path = "DecAstar/"


def run_episode(agent, M, N, exp_name, T_threshold, sample=True, scenario_dir=None, VISUALIZE=False, heuristic=False,
                n_sample=1):
    '''
    해당 함수에서는 한 episode를 돌리는 것인데, 하나의 policy를 이용해서, 각 agent에 대해 task를 assign해줌.
    그리고 수행을 하는데 있어서 threshold보다 높게 남은 것은 다시 reschdule해주는 방식임.
    '''
    actions = [[] for _ in range(M)] #! 이게 각 agent마다 바로 다음에 끝날 task의 idx넣는 거라 range가 num of agent가 되야함.
    scenario = load_scenarios(scenario_dir) #* map, agent, task 위치 assign된 것.
    task_finished_bef = np.array([False for _ in range(N)]) #* task마다 complete check array인듯.
    grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
    save_map(grid, exp_name)
    shortest_paths = compute_astar(agent_pos, total_tasks, graph) 
    #* 각 task에서 각 agent까지의 shortest path dist list로 가지고 있음.
    #* shortest_paths[i][j] 하면 i task에서 j agent까지의 shortest path.

    itr = 0
    episode_timestep = 0

    # `task_finished` defined for each episode
    g, ag_node_indices, task_node_indices = convert_dgl(graph, agent_pos, total_tasks, task_finished_bef,
                                                        shortest_paths)
    #* agent, task로 이루어진 bipartite graph, 기존 graph에서 agent node의 idx , task의 idx
    joint_action_prev = np.array([0] * M) 
    ag_order = np.arange(M) # 아마 priority
    continuing_ag = np.array([False for _ in range(M)])

    while True:
        """ 1.super-agent coordinates agent&task pairs """
        # `task_selected` initialized as the `task_finished` to jointly select task at each event
        best_T = None
        best_ordered_joint_action = None
        best_agent_traj = None
        best_curr_tasks = None
        best_joint_action = None
        max_T = 100

        for _ in range(n_sample): #* n번 assign을 실행하고 trajectory를 구해서, max trjectory 길이가 가장 작은 task assign을 고르는 것.
            if itr > 1:
                random.shuffle(remaining_ag_idx)
                ag_order = np.array(continuing_ag_idx + remaining_ag_idx)

            curr_tasks = [[] for _ in range(M)]  # in order of init agent idx
            if heuristic:
                joint_action = agent.forward_heuristic(g, ag_order, continuing_ag, joint_action_prev, sample=sample)
            else:
                joint_action = agent(g, ag_order, continuing_ag, joint_action_prev, sample=sample)
                #! ValueError: Expected parameter probs (Tensor of shape (21,)) of distribution Categorical(probs: torch.Size([21])) to satisfy the constraint Simplex(), but found invalid values:
                #!    tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                #!    grad_fn=<DivBackward0>)
            #* 결국 agent를 통해서 나오는 output은 각 agent가 선택한 task list. agent를 index로 해서 value가 task index.
            ordered_joint_action = [0] * M

            # convert action to solver format
            # TODO: batch

            for ag_idx, action in zip(ag_order, joint_action):
                if action < N:
                    task_loc = g.nodes[action + M].data['original_loc'].squeeze().tolist()
                else: #? dummy task를 의미하나 즉, 휴식?
                    task_loc = agent_pos[ag_idx].tolist() #* dummy 선택하면 본인 위치 선택.

                curr_tasks[ag_idx] = [task_loc] #* 수행할 task의 좌표.
                ordered_joint_action[ag_idx] = action #* 선택한 task index.

            """ 2.pass created agent-task pairs to low level solver """
            # convert action to the solver input formation
            #* 현재 task assign대로 eecbsfmf dnlgks .scen파일 작성. #*위에서 정한 task대로 low level solver실행.
            save_scenario(agent_pos, curr_tasks, exp_name, grid.shape[0], grid.shape[1])

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

            # process_out.stdout format
            # runtime, num_restarts, num_expanded, num_generated, solution_cost, min_sum_of_costs, avg_path_length
            process_out = subprocess.run(c, capture_output=True)
            text_byte = process_out.stdout.decode('utf-8')
            try:
                sum_costs = int(text_byte.split('Succeed,')[-1].split(',')[-3])
            except:
                agent.replay_memory.memory = []
                return None, itr

            #!
            # if itr > N: #* 이렇게 구현을 해놓은 것이, 전부 다 dummy로 들어가서 쉬는 상황이 없으면 최대 task의 갯수 만큼만 assign해주면 된다는 생각일텐데, 전부 dummy로 들어가는 경우가 생긴다는 것이 문제.
            #     return None, itr

            # Read solver output
            agent_traj = read_trajectory(solver_path + exp_name + "_paths.txt")
            T = np.array([len(t) for t in agent_traj]) #각 agent 별 trajectory length

            if max_T > max(T): #* sampling을 여러번 해서 best trajectory 얻음.
                #! ValueError: max() arg is an empty sequence
                best_T = T
                best_curr_tasks = curr_tasks
                best_agent_traj = agent_traj
                best_ordered_joint_action = ordered_joint_action # agent idx에는 agent가 선택한 task idx있음.
                max_T = max(T)
                best_joint_action = joint_action

        # TODO makespan -> sum of cost
        # TODO batch training loop

        # Mark finished agent, finished task
        #! 아래 코드에서 ValueError: zero-size array to reduction operation minimum which has no identity
        #! array가 빌 경우는 다음 step에 task가 다 끝나는 경우임.
        # next_t = best_T[best_T > 1].min() #* 각 agent의 step 수 중 1 초과인 것만 남기고 그중 min. 아 trajectory가 시작 노드부터 있어서 무조건 1임. 그래서 다음 step에 끝나는 것은 길이가 2
        #* 가장 빠른 다음 끝나는 agent면서 현재 하는 task가 올바른(N이하인) 것.
        ###
        filtered_T = best_T[best_T > 1]
        if filtered_T.size > 0:
            print(f"T : {best_T}")
            next_t = filtered_T.min()
        else: #* 전부 다 dummy task에 assign된 경우.
            # print(f"finished_ag : {finished_ag}")
            print(f"errored T : {best_T}")
            next_t = 1
            # next_t = 1000 #* 이렇게 안끝날 경우 즉, 전부 쉴경우는 생기면 안되니까 무조건 penalty를 주는 것은?
            # agent.push(g, best_ordered_joint_action, _, _, next_t, True)
            # episode_timestep += next_t - 1
            # return episode_timestep, itr
        # if itr > 100: #policy 고를 때, 1e-5를  더해주도록 구현해서 무한으로 가진 않음.
        #     next_t = 1000 
        #     agent.push(g, best_ordered_joint_action, _, _, next_t, True)
        #     episode_timestep += next_t - 1
        #     return episode_timestep, itr
        ###

        #* 아래 array는 다음 끝나는 agent flag.
        finished_ag = (best_T == next_t) * (
                np.array(best_ordered_joint_action) < N)  # as more than one agent may finish at a time #*아 agent로 도축된 action부터는 task가 0~N-1까지의 index임.
        #! 아래 부분에서 dummy에 assign되는 경우, finished가 되지 않음. 모든 agent가 dummy로 가는 경우 task를 받을 agent가 없음.
        #! 그래서 error나는 것이라 생각하고 수정
        ###
        # finished_ag = (best_T == next_t)
        # finished_ag = (best_T <= next_t) #! 이렇게 하면 dummy에 assign된 것 때문에 오류남 dummy가 애초에 왜 필요하지??? stay인가? 그런듯..
        ###
        finished_task_idx = np.array(best_ordered_joint_action)[finished_ag] #* 다음 끝나는 task
        task_finished_aft = deepcopy(task_finished_bef)
        task_finished_aft[finished_task_idx] = True #* 끝나는 task표시.
        episode_timestep += next_t - 1 #*원래 기본 trajectory가 1이니까 -1step 후에 끝나는 것.

        for i, _fin in enumerate(finished_ag):
            if _fin: #* true일 경우
                actions[i].append(best_ordered_joint_action[i]) #* action에 다음에 끝나는 task담음.

        # overwrite output
        agent_pos_new = deepcopy(agent_pos) #* 현재 agent position.
        for ag_idx in ag_order:
            if best_T[ag_idx] > 1: #* best_T는 agent idx에 대한 trajectory length. #* dummy로 인해 stay가 아닌 agent에 대해 다음 위치 assign.
                agent_pos_new[ag_idx] = best_agent_traj[ag_idx][next_t - 1] #* 가장빨리 끝나는 다음 step에 각 agent의 위치.

        # Replay memory 에 transition 저장. Agent position 을 graph 의 node 형태로
        terminated = all(task_finished_aft) #* 모든 task가 finished인지 check.

        # TODO: training detail
        if sample: #* score를 prob삼아서 sampling을 통해 action을 정한 경우. replay_mem에 저장. training에서는 tr
            agent.push(g, best_ordered_joint_action, ag_order, deepcopy(task_finished_bef), next_t, terminated)
        #* bipartite graph, 현재 agent별 assigned task, 아마 priority, 이전 task_finished정보, 바로 다음 task끝나는 step, 종료 정보 를 buffer에 담음.
        if VISUALIZE:
            # visualize
            vis_ta(graph, agent_pos, best_curr_tasks, str(itr) + "_assigned", total_tasks=total_tasks,
                   task_finished=task_finished_bef)
            vis_ta(graph, agent_pos_new, best_curr_tasks, str(itr) + "_finished", total_tasks=total_tasks,
                   task_finished=task_finished_aft)

        if terminated: #* 모든 task finished
            print(actions) #* action은 이제까지 각 agent에 assign된 task.
            return episode_timestep, itr

        # agent with small T maintains previous action
        continuing_ag = (0 < best_T - next_t) * (best_T - next_t < T_threshold) 
        #* 가장 빠른 다음 task종료 후에도 trajectory움직여야 하는 것 중, threshold step보다는 적게 남은 것은 계속 수행. 아닌 것은 rescheduling.
        continuing_ag_idx = continuing_ag.nonzero()[0].tolist() #* 계속 움직이는 것 list로 만듬.
        remaining_ag = list(set(range(M)) - set(continuing_ag_idx)) #* rescheduling 해줘야 하는 agent index.
        #* 즉, 이번에 끝난 agent도, dummy에 들어가서 쉰 agent도 remaining_ag에 들어감.

        # option 1. randomly select remaining ag
        if RANDOM: #* priority random해주는 거인듯.
            random.shuffle(remaining_ag)
            remaining_ag_idx = remaining_ag
        else: #* priority를 task dist가 가까운 순서로 해주는 것인듯. #? 왜 가까운거부터 해주지?
            # option 2. sort remaining ag by remaining task dist
            dists = g.edata['dist'].reshape(-1, M).T
            finished = task_finished_aft.nonzero()[0] # 끝난 task index.
            reserved = np.array(best_joint_action)[continuing_ag_idx] #각 agent별 선택된 task.
            dists[:, finished] = 9
            dists[:, reserved] = 9

            # option 2-1 sort by mean
            remaining_ag_dist = dists[remaining_ag].mean(-1)
            # option 2-2 sort by min
            # remaining_ag_dist = dists[remaining_ag, :-1].min(-1).values

            remaining_order = remaining_ag_dist.sort().indices
            remaining_ag_idx = np.array(remaining_ag)[remaining_order].tolist()

        if type(remaining_ag_idx) == int:
            remaining_ag_idx = [remaining_ag_idx]
        # ========================
        ag_order = np.array(continuing_ag_idx + remaining_ag_idx) #* reschedule해야하는 것을 현재 진행중인 것 뒤에 붙임.
        assert len(set(ag_order)) == M
        joint_action_prev = np.array(best_ordered_joint_action, dtype=int)

        agent_pos = agent_pos_new #* 가장 빨리 끝나는 다음task episode time에서의 agent들의 위치. #* update된 agent 위치.
        task_finished_bef = task_finished_aft #* 다음 time에 끝나는 것들.
        shortest_paths = compute_astar(agent_pos, total_tasks, graph) #* 각 agent의 현재 위치에서 모든 task로의 A* 재계산.
        g, ag_node_indices, _ = convert_dgl(graph, agent_pos, total_tasks, task_finished_bef, shortest_paths)
        #* bipartite graph를 만들고, node feature, edge feature도 넣어줌. 논문에서의 그대로
        itr += 1


if __name__ == '__main__':
    from tqdm import tqdm
    #! main_nn.py를 오류 없이 실행하기 위해서는 아래 sample_per_epoch의 수 만큼의 scenario가 존재하는지 확인하고
    #! _eval scenario또한 그 수 만큼 존재하는지 확인해줘야힘.
    epoch = 1000
    sample_per_epoch = 1 #! 이것도 senario 몇개 만들었는지에 따라서 조절 해줘야함.

    M, N = 10, 20
    # M, N = 20, 50
    n_sample = 1
    #! threshold 처리 logic에 문제가 있는 듯. 작게 설정했을 때, infinite loop빠지는 경우 생김.
    T_threshold = 150  # N step fwd #*이정도 step 앞에서도 task가 종료되지 않는 경우, 다시 scheduling하겠다는 것.
    agent = Agent() #* 이렇게 하면 difault gnn layer 1
    # agent.load_state_dict(torch.load('saved/20230220_1424.th'))
    n_eval = 1 #* max score로 action 즉, task를 정하니까 한번만 수행하면 되는 듯.
    best_perf = 1000000
    exp_name = datetime.now().strftime("%Y%m%d_%H%M")
    wandb.init(project='etri-mapf', group=f"{M}_{N}_nsam_{n_sample}_spe_{sample_per_epoch}", name=f"{M}_{N}_{n_sample}_{sample_per_epoch}_{exp_name}")
    
    if not os.path.exists('saved'):
        os.makedirs('saved')

    for e in range(epoch):
        epoch_perf = []
        epoch_loss = []
        epoch_itr = []
        eval_performance = []
        eval_itr = []

        # train.
        for sample_idx in tqdm(range(sample_per_epoch)):
            scenario_dir = '323220_1_{}_{}/scenario_{}.pkl'.format(M, N, sample_idx + 1)
            episode_timestep, itr = run_episode(agent, M, N, exp_name, T_threshold, sample=True,
                                                scenario_dir=scenario_dir)
            print(f"train run episode return : {episode_timestep}, {itr}")
            if episode_timestep is not None: #* episode terminate 시.
                fit_res = agent.fit() #* step.
                epoch_perf.append(episode_timestep) #* 총 걸린 시간이 makespan이니까.
                epoch_itr.append(itr) #* 몇 번 scheduling에 들어갔는지.
                epoch_loss = fit_res['loss'] #* 학습 loss.

        # evaluation.
        for i in tqdm(range(n_eval)):
            scenario_dir = '323220_1_{}_{}_eval/scenario_{}.pkl'.format(M, N, i + 1)
            episode_timestep, itr = run_episode(agent, M, N, exp_name, T_threshold, sample=False,
                                                scenario_dir=scenario_dir) #* max score를 가지는 task assign을 고름.
            print(f"eval run episode return : {episode_timestep}, {itr}")
            if episode_timestep is not None:
                eval_performance.append(episode_timestep)
                eval_itr.append(itr)

        wandb.log({'epoch_loss_mean': np.mean(epoch_loss), # episodes 진행한 것에 대한 loss.
                   'epoch_cost_mean': np.mean(epoch_perf), # episodes 진행한 것에 대한 tot_cost. makespan
                   'e': e, # 몇 epoch째인지.
                   'eval_cost_mean': np.mean(eval_performance),
                   'epoch_itr_mean': np.mean(epoch_itr), # 얼마나 scheduling이 일어났는지 terminate까지.
                   'eval_itr_mean': np.mean(eval_itr),
                   'n_sample': len(epoch_itr)})

        if np.mean(eval_performance) < best_perf:
            torch.save(agent.state_dict(), 'saved/{}.th'.format(exp_name))
            best_perf = np.mean(eval_performance)
