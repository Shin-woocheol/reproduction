import random

import numpy as np

from utils.astar import graph_astar

def removal(task_status, tasks, graph, N=2):
    # incomplete task만 골라줌.
    incomplete_tasks = [i for i, st in enumerate(task_status) if st == 0]
    if len(incomplete_tasks) == 0:
        return []

    chosen = random.choice(incomplete_tasks)
    incomplete_tasks.remove(chosen)

    if len(incomplete_tasks) <= N: # 추가로 제거해야할 수 보다 작은경우 나머지 다 제거
        removal_idx = [chosen] + incomplete_tasks
        return removal_idx

    rs = {}
    for r in incomplete_tasks:
        rs[r] = relatedness(graph, tasks[chosen], tasks[r])
    # rs: { task_id : relatedness_value }

    sorted_r = sorted(rs.keys(), key=lambda x: rs[x], reverse=True)
    additional = sorted_r[:N]

    removal_idx = [chosen] + additional
    return removal_idx

def relatedness(graph, ti, tj, w1=9, w2=3):
    _, d_si_sj = graph_astar(graph, ti[0], tj[0])
    _, d_gi_gj = graph_astar(graph, ti[-1], tj[-1])
    return w1 * (d_si_sj + d_gi_gj) + w2 * (len(ti) + len(tj))

#* 현재 문제 상황에서는 start와 goal이 같은 상황. pickup and delivery가 아님.
# def relatedness(graph, ti, tj, w1=9, w2=3):
#     _, d_si_sj = graph_astar(graph, ti[0], tj[0])
#     return w1 * (d_si_sj) + w2 * (len(ti) + len(tj))