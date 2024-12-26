import time

from LNS.hungarian import hungarian
from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from utils.generate_scenarios import load_scenarios
from utils.generate_scenarios import save_scenarios
from utils.soc_ms import cost
from utils.vis_graph import vis_dist, vis_ta

"""
Create random scenarios and load one of them
"""
M = 10
N = 20
# save_scenarios(itr = 20, M=20, N=50) #* generate a new map and make scenario itr times
scenario = load_scenarios(f'323220_1_{M}_{N}_eval/scenario_1.pkl') #* size,size,obs_density,tasklength,agent,task
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
total_tasks = [[[x, y]] for x, y in total_tasks]
#* array map, graph, agent pos list[[]], task list[[[]]]
vis_dist(graph, agent_pos, total_tasks) #* fig folder에 scenario 그림 존재.

"""
1st step: Hungarian Assignment
"""
h_time = time.time()
task_idx, tasks = hungarian(graph, agent_pos, total_tasks) #* hungarian algorithm으로 초기 assign. 
#* task assign dictionary와 h_tasks는 dict인데 안에 좌표 담고있음.
'''
task_idx : {0: [34, 1, 45], 1: [43, 5], ...
h_tasks : {0: [{34: [[2, 3]]}, {1: [[1, 7]]}, {45: [[4, 25]]}], 1: [{43: [[12, 4]]}, {5: [[12, 3]]}], ...
'''
h_time = time.time() - h_time
soc, ms = cost(agent_pos, tasks, graph) #* 여기서 받은 cost는 주변 agent와 연관 없이 A*로만 받은 것.
#* 그래서 agent_cost_list의 sum을 해주면 모든 agent가 task 수행시 움직이는 step의 합 -> soc
#* max를 해서 넘기면 task all solve까지 걸리는 시간 -> ms
print('INIT || SOC: {:.4f} / MAKESPAN: {:.4f} / TIMECOST: {:.4f}'.format(soc, ms, h_time))

vis_ta(graph, agent_pos, tasks, 'HA') #* assign된 것 fig folder에 있음. 이건 ta_HA에.

"""
2nd step: Large Neighborhood Search (iteratively) 로 task 다시 assign하면서 그림.
sum of cost 떨어지는 방향으로 찾음. 시간이 지날수록 성능 좋음.
"""
max_t = time.time() + 10000  # time limit: 10s
itr = 0

while True:
    lns_time = time.time()
    # Destroy
    removal_idx = removal(task_idx, total_tasks, graph, N=2)
    for i, t in enumerate(tasks.values()):
        for r in removal_idx:
            if {r: total_tasks[r]} in t:
                tasks[i].remove({r: total_tasks[r]})
                
    # Reconstruct
    while len(removal_idx) != 0:
        f = f_ijk(tasks, agent_pos, removal_idx, total_tasks, graph)
        regret = get_regret(f)
        regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
        re_ins = list(regret.keys())[0]
        re_a, re_j = regret[re_ins][1], regret[re_ins][2]
        removal_idx.remove(re_ins)
        to_insert = {re_ins: total_tasks[re_ins]}
        tasks[re_a].insert(re_j, to_insert)

    lns_time = time.time() - lns_time
    if time.time() > max_t:
        break

    itr += 1
    soc, ms = cost(agent_pos, tasks, graph)
    print('{}_Solution || SOC: {:.4f} / MAKESPAN: {:.4f} / TIMECOST: {:.4f}'.format(itr, soc, ms, lns_time))
    vis_ta(graph, agent_pos, tasks, itr)

# TODO [1] removal parameter N test / [2] time consuming regret search
