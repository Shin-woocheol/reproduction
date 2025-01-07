from copy import deepcopy
from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from utils.soc_ms import cost

def lns_once(current_tasks, agent_pos, total_tasks, graph, task_status, init_agent_pos, N=2):
    # Destroy
    #* shaw removal. relatedness를 기반으로 random으로 정해진 remove task와 가장 관련된 task N개를 추가로 삭제.
    old_tasks = deepcopy(current_tasks)
    removal_idx = removal(task_status, total_tasks, graph, N=N)

    #* tasks에서 removal tasks 제거.
    for i, t in enumerate(current_tasks.values()): 
        for r in removal_idx:
            if {r: total_tasks[r]} in t:
                current_tasks[i].remove({r: total_tasks[r]})
                
    # Reconstruct
    #* regret based re-insertion
    while len(removal_idx) != 0:
        f = f_ijk(current_tasks, agent_pos, removal_idx, total_tasks, graph, task_status) #* estimated total service time when inserting task at the jth position of task sequence of agent k
        regret = get_regret(f)
        regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
        re_ins = list(regret.keys())[0]
        re_a, re_j = regret[re_ins][1], regret[re_ins][2]
        removal_idx.remove(re_ins)
        to_insert = {re_ins: total_tasks[re_ins]}
        current_tasks[re_a].insert(re_j, to_insert)
    
    # 새롭게 assign한 task seq의 make span이 더 작거나 같은 경우에만 update
    if cost(init_agent_pos, current_tasks, graph)[1] > cost(init_agent_pos, old_tasks, graph)[1]:
        return old_tasks
    # print("task changed")
    return current_tasks