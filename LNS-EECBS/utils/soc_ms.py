from utils.astar import graph_astar
import subprocess
import numpy as np
from copy import deepcopy
from math import inf
from utils.solver_util import save_map, save_scenario, read_trajectory, compute_astar

solver_path = "EECBS/"
dec_solver_path = "DecAstar/"

def cost(agent_pos, solution, graph):
    '''
    agent_pos : agent position list
    solution : agent별 assign된 task와 좌표
    graph : grid graph.
    '''
    #* 이 코드에서는 각 agent에 할당된 task를 순회하는데에 걸리는 step을 모두 계산함.
    #* 그래서 agent_cost_list의 sum을 해주면 모든 agent가 task 수행시 움직이는 step의 합
    #* max를 해서 넘기면 task all solve까지 걸리는 시간
    #* solution은 agent에 할당된 task와 task의 좌표.
    # h_tasks : {0: [{34: [[2, 3]]}, {1: [[1, 7]]}, {45: [[4, 25]]}],
    agent_cost_list = list()
    for i in solution:
        path = list()
        agent_cost = 0 if len(solution[i]) == 0 else graph_astar(graph, agent_pos[i],
                                                                 list(solution[i][0].values())[0][0])[1]
        for a in solution[i]:
            for b in a.values():
                path += b
        for s, g in zip(path[:-1], path[1:]):
            agent_cost += graph_astar(graph, s, g)[1]
        agent_cost_list.append(agent_cost)

    return sum(agent_cost_list), max(agent_cost_list)

def eecbs_cost(agent_pos, tasks, total_tasks, exp_name, size):
    tasks = deepcopy(tasks)
    agent_pos = deepcopy(agent_pos)
    M = len(agent_pos)     
    make_span = 0 

    while True:
        curr_tasks_pos = [[] for _ in range(M)]  # 각 agent가 이번 EECBS에서 수행할 목적지(1개)
        for i in range(M):
            if len(tasks[i]) > 0:
                first_task_dict = tasks[i][0]
                for coords_list in first_task_dict.values():
                    curr_tasks_pos[i] = coords_list 
            else: # 해당 agent에게 할당된 task 전부 수행 완료.
                curr_tasks_pos[i] = [agent_pos[i]]

        save_scenario(agent_pos, curr_tasks_pos, exp_name, size, size)
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

        moving_agents_mask = (T > 1)
        if not np.any(moving_agents_mask): # 전부 다 제자리인 경우. 즉, 남은 task 없음.
            break

        next_t = T[moving_agents_mask].min()
        make_span += (next_t - 1)

        finished_ag = (T == next_t)

        new_agent_pos = deepcopy(agent_pos)
        for i in range(M):
            if T[i] >= 2: #이동
                new_agent_pos[i] = agent_traj[i][next_t - 1]
            else: #제자리
                new_agent_pos[i] = agent_pos[i]

        for i in range(M):
            if finished_ag[i] and len(tasks[i]) > 0:
                tasks[i].pop(0)
        agent_pos = new_agent_pos

    return make_span
