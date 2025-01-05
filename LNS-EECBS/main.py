import time

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
exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
save_map(grid, exp_name)

vis_dist(graph, agent_pos, total_tasks) #* fig folder에 scenario 그림 존재.

"""
1st step: Hungarian Assignment
#* hungarian algorithm으로 초기 assign. 
"""
h_time = time.time()
task_idx, tasks = hungarian(graph, agent_pos, total_tasks) 
#* task assign dictionary와 h_tasks는 dict인데 안에 좌표 담고있음.
'''
task_idx : {0: [34, 1, 45], 1: [43, 5], ...
h_tasks(tasks) : {0: [{34: [[2, 3]]}, {1: [[1, 7]]}, {45: [[4, 25]]}], 1: [{43: [[12, 4]]}, {5: [[12, 3]]}], ...
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
max_itr = 10000
itr = 0

while True:
    lns_time = time.time()
    tasks = lns_once(tasks, task_idx, agent_pos, total_tasks, graph, N = 2)
    lns_time = time.time() - lns_time
    # if time.time() > max_t:
    #     break
    if itr > max_itr:
        break

    itr += 1
    soc, ms = cost(agent_pos, tasks, graph)
    make_span = eecbs_cost(agent_pos, tasks, total_tasks, exp_name, grid.shape[0])
    print(f'{itr}_Solution || EECBS_MAKESPAN : {make_span} / SOC: {soc} / MAKESPAN: {ms} / TIMECOST: {lns_time:.4f}')
    if itr % 100 == 0:
        vis_ta(graph, agent_pos, tasks, itr)

# TODO [1] removal parameter N test / [2] time consuming regret search
#* LNS를 사용해서 얻은 task assign을 EECBS로 어떻게든 넘겨주고 EECBS로 path찾아서 cost만 반환받으면 될듯.
#* soc가 아니라, makespan을 줄이도록 할 수 있는 방법은??
#* 해당 알고리즘이 다시 돌아가는 즉, task assign이 다시 일어나는 시점은, 어떤 agent가 처음으로 종료되는 시점.
#* 해당 시점으로 agent의 position update.
#* 그럼 다시 task assign후에 다시 EECBSfmf ehfflsms qkdtlr.