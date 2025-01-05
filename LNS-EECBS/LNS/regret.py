from copy import deepcopy

import numpy as np
import random

from utils.astar import graph_astar


def f_ijk(current_tasks, agent_pos, removal_idx, total_tasks, graph):
    '''
    current_tasks : task removed된 상태에서의 task assignment dict. 
                    key : agent_idx value : dict(key : task idx, value : position)
    agent_pos : agent의 현재 position list
    removal_idx : removed task idx list
    total_tasks : 각 task idx별 좌표 list
    graph : grid graph

    estimated total service time when inserting task at the jth position of task sequence of agent k
    '''
    before_cost = list() #* agent idx에 task종료까지의 cost list. task 삭제된 후 계산
    for b in current_tasks.keys(): #agent
        b_path = list()
        for _a in current_tasks[b]: #task
            for _b in _a.values(): #task positions
                b_path += _b
        b_cost = 0 if len(b_path) == 0 else graph_astar(graph, agent_pos[b], b_path[0])[1] #agent 첫 pos to 첫 task
        for _s, _g in zip(b_path[:-1], b_path[1:]): #first to last-1, second to last #* task seq 사이의 길이 계산.
            b_cost += graph_astar(graph, _s, _g)[1]
        before_cost.append(b_cost) #* 해당 agent의 task 종료까지의 shortest path 합.

    f = dict(zip(removal_idx, [list() for _ in range(len(removal_idx))])) #각 removed task idx별 하나의 list
    #* removal task에 대해서, agent의 수만큼 list 그 list에는 각 position에 넣었을 때의 cost.
    #* remove된 task를 모든 가능한 위치에 넣어보면서 cost를 구함. 
    #TODO cost가 해당 agent의 cost인지, 전체 agent의 cost인지를 봐야함.
    for i in removal_idx:
        removed_task_pos = total_tasks[i]
        for k in range(len(agent_pos)):
            agent_task_seq = current_tasks[k]
            f_list = list()
            for j in range(len(agent_task_seq) + 1):
                temp_seq = deepcopy(agent_task_seq)
                temp_seq.insert(j, {i: removed_task_pos}) #첨부터 끝까지 넣어봄.
                # cost via A* after re-insertion i, j, k
                path = list()
                for a in temp_seq:
                    for b in a.values(): #position
                        path += b

                #* 바뀐 task seq에서 cost계산.
                f_value = graph_astar(graph, agent_pos[k], path[0])[1]
                for s, g in zip(path[:-1], path[1:]):
                    f_value += graph_astar(graph, s, g)[1]

                #* option1. minimize soc
                # for o in current_tasks.keys() - [k]:
                #     f_value += before_cost[o]
                # f_list.append(f_value)

                #* option2. minimize makespan
                cost = []
                cost.append(f_value)
                for o in current_tasks.keys():
                        cost.append(before_cost[o])
                makespan_val = max(cost)
                f_list.append(makespan_val)

            f[i].append(f_list)
    return f #* removal task 각각에 대해서, 해당 task를 k번째 agent의 j위치에 넣었을 때의 estimated time을 list으로 넣음.

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
