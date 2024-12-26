import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.astar import graph_astar


def cost_matrix(g, a, t): #* 각 agent에서 task로의 a* 계산.
    """
    t : total_tasks

    각 agent to task까지의 A* distance
    """
    m = np.zeros((len(a), len(t)))
    for i in range(len(a)):
        for j in range(len(t)):
            m[i][j] = graph_astar(g, a[i], t[j][0])[1] #* graph a*에 agent, task position을 넣음.
            #아래 정보가 기록됨. 근데 [1]이니까 path cost만.
#                  path : [(28, 23), (28, 22), (28, 21), (28, 20), (28, 19)]  #node to node
#                   path_cost : 4
    return m


def hungarian(graph, agent_pos, tasks):
    '''
    graph : grid graph
    agent_pos : agent pos index
    tasks : task pos index

    return
    task_idx : dict, key = agent idx, value = task idx
    h_tasks : dict, key = agent idx, value dict that task idx and its position.
    '''
    # print(tasks)
    cm = cost_matrix(graph, agent_pos, tasks)
    ag, assigned = linear_sum_assignment(cm) #* 일단 agent당 1개씩 assign
    # print(f"hungarian test \n ag : {ag}, assigned : {assigned}")
    task_idx = dict(zip(ag, assigned))
    tasks_idx = list(range(len(tasks)))
    unassigned_idx = list(set(tasks_idx) - set(assigned))
    unassigned = [tasks[ut] for ut in unassigned_idx]

    first = True
    while len(unassigned) != 0:
        if first:
            na = [tasks[t][-1] for t in task_idx.values()]
        else:
            na = [tasks[t[-1]][-1] for t in task_idx.values()]
        cm = cost_matrix(graph, na, unassigned)
        ag, assigned = linear_sum_assignment(cm)
        assigned = [unassigned_idx[t_idx] for t_idx in assigned]
        unassigned_idx = list(set(unassigned_idx) - set(assigned))
        for a, t in zip(ag, assigned):
            if type(task_idx[a]) == np.int64:
                task_idx[a] = [task_idx[a]] + [t]
                unassigned.remove(tasks[t])
            else:
                task_idx[a].append(t)
                unassigned.remove(tasks[t])
        first = False

    h_tasks = dict()
    for k in task_idx.keys():
        # h_tasks[k] = [{'s': [agent_pos[k].tolist()]}]
        if type(list(task_idx.values())[k]) == np.int64:
            i = list(task_idx.values())[k]
            h_tasks[k] = [{i: tasks[i]}]
        else:
            h_tasks[k] = [{i: tasks[i]} for i in list(task_idx.values())[k]]

    # print(f"hungarian test \n task_idx : {task_idx}, h_tasks : {h_tasks}")
    #* task assign dictionary와 h_tasks는 dict인데 안에 좌표 담고있음.
    '''
    task_idx : {0: [34, 1, 45], 1: [43, 5], ...
    h_tasks : {0: [{34: [[2, 3]]}, {1: [[1, 7]]}, {45: [[4, 25]]}], 1: [{43: [[12, 4]]}, {5: [[12, 3]]}], ...
    '''
    return task_idx, h_tasks
