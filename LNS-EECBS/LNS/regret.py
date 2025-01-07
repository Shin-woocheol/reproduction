from copy import deepcopy

import numpy as np
import random

from utils.astar import graph_astar

def f_ijk(current_tasks, agent_pos, removal_idx, total_tasks, graph, task_status):
    """
    remove된 task 각각을 task seq 각 사이에 넣어보면서 cost계산한 dict를 return
    """
    before_cost = [] # removed된 task seq에서의 cost
    for agent_id in current_tasks.keys():
        path_for_agent = []
        for t_dict in current_tasks[agent_id]:
            for t_id, coords in t_dict.items():
                # status가 0 or 1 인 task만 경로에 추가
                if task_status[t_id] in (0,1): #incomplete, executing task
                    path_for_agent += coords

        # path_for_agent를 따라 A* 거리 계산
        if len(path_for_agent) == 0:
            b_cost = 0
        else:
            # agent 현재 위치 -> 첫 task
            b_cost = graph_astar(graph, agent_pos[agent_id], path_for_agent[0])[1]
            # task 간 이동
            for s, g in zip(path_for_agent[:-1], path_for_agent[1:]):
                b_cost += graph_astar(graph, s, g)[1]
        before_cost.append(b_cost)

    #  f : {removed_task : [ [agent0: cost들], [agent1: cost들], ... ] }
    f = dict(zip(removal_idx, [list() for _ in range(len(removal_idx))]))

    for r_task_id in removal_idx:
        r_coords = total_tasks[r_task_id]
        for k in range(len(agent_pos)):
            agent_task_seq = current_tasks[k]
            cost_list_for_agent = []
            for j in range(len(agent_task_seq) + 1):
                temp_seq = deepcopy(agent_task_seq)
                temp_seq.insert(j, {r_task_id: r_coords})

                if j < len(temp_seq) - 1: #마지막 제외
                    next_task_dict = temp_seq[j+1]
                    n_tid = list(next_task_dict.keys())[0]
                    if task_status[n_tid] in (1,2): #complete, executing이전에 넣으면 안됨.
                        cost_list_for_agent.append(float('inf'))
                        continue

                path = []
                for t_dict2 in temp_seq:
                    for tid2, coords2 in t_dict2.items():
                        path += coords2

                if len(path) == 0:
                    f_value = 0
                else:
                    f_value = graph_astar(graph, agent_pos[k], path[0])[1]
                    for s, g in zip(path[:-1], path[1:]):
                        f_value += graph_astar(graph, s, g)[1]

                # make span 계산
                cost_candidates = []
                for o in range(len(agent_pos)):
                    if o == k:
                        cost_candidates.append(f_value)
                    else:
                        cost_candidates.append(before_cost[o])

                makespan_val = max(cost_candidates)
                cost_list_for_agent.append(makespan_val)

            f[r_task_id].append(cost_list_for_agent)

    return f


def random_argmin(v):
    """
    v에서 최소값을 갖는 모든 인덱스 중 하나를 랜덤으로 반환
    """
    arr = np.array(v)
    min_value = arr.min()
    candidates = np.where(arr == min_value)[0]  # 최소값을 갖는 인덱스들의 리스트
    return random.choice(candidates.tolist()) 

def get_regret(f_values):
    #* regret은 첫번째로 가장 짧은 것과 두 번째로 가장 짧은 것의 차이.
    regret = dict()
    for k, vs in f_values.items(): # k: removed idx, vs : agents 각 위치넣었을 때 cost
        v = list() # 각 agent에서 각 위치에 넣었을 때의 cost를 flatten 해서 넣어줌.
        for v_id in range(len(vs)):
            v += vs[v_id] # list 풀어서 넣어줌. 
        # temp = np.argmin(v) + 1 # 전체 최소 cost idx + 1
        temp = random_argmin(v) + 1
        #* 전체 최소 cost 가진 agent idx찾기
        for a, v_ in enumerate(vs):
            temp -= len(v_)
            if temp <= 0:
                break

        j = np.argmin(vs[a]) #전체 최소를 가진 agent의 cost list에서 최소 cost idx
        v = sorted(v) #전체 cost 정렬
        regret[k] = [v[1] - v[0], a, j] # removed idx에 regret, 최소 agent idx, cost 줌.
    return regret
