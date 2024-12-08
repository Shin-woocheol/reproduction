import os
import subprocess

from nn.agent import Agent
from nn.ag_util import embed_traj
from utils.generate_scenarios import load_scenarios, save_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory
from utils.vis_graph import vis_dist

#* low level path finding algorithm

solver_path = "EECBS/"
# M, N = 10, 10
#* num of agent, task
M, N = 20, 50 
# if not os.path.exists('scenarios/323220_1_{}_{}/'.format(M, N)):
# save_scenarios(size=32, M=M, N=N)

scenario = load_scenarios('323220_1_{}_{}/scenario_1.pkl'.format(M, N))
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
#* array map, graph, agent pos list[[]], task list[[[]]] # task assign안된상태. 
'''
agent_pos : [[ 0  8]
 [11 15]...
total tasks : [[[23, 2]], [[25, 5]], ...
'''
vis_dist(graph, agent_pos, total_tasks)

scenario_name = 'test1'
save_map(grid, scenario_name) #* eecbs사용을 위한 map으로 변경. .map을 만듬.
total_tasks_bef = total_tasks

ag = Agent()

# TODO total_tasks format: 태스크 전체 (순서대로) 쭉 나열된 일종의 task 집합 -> 'agent to task' 할당 집합으로 바꿔야 함 (main.py에서 tasks에 해당)

for i in range(10):
    #* eecbs에 사용될 .scen만드는 함수.
    save_scenario(agent_pos, total_tasks_bef, scenario_name, grid.shape[0], grid.shape[1]) #* 위 save_scenarios()와는 다른 함수.
    #* agent의 index와 같은 task를 일단 가지도록 설정되어있음.
    c = [solver_path + "eecbs",
         "-m",
         solver_path + scenario_name + '.map',
         "-a",
         solver_path + scenario_name + '.scen',
         "-o",
         solver_path + scenario_name + ".csv",
         "--outputPaths",
         solver_path + scenario_name + "_paths.txt",
         "-k", "{}".format(M), "-t", "60", "--suboptimality=1.2"]
    subprocess.run(c) #* EECBS폴더에 task로의 paths.txt파일 생성됨.
    agent_traj = read_trajectory(solver_path + scenario_name + "_paths.txt")
    #* agent_traj에는 각 agent의 index대로, eecbs로 생성한 trajectory에 대해 list의 형태로 들어가있음.
     #print(f"agent_traj : {agent_traj}")/
    # [..., [[29, 13], [29, 12], [29, 11], [29, 10], [29, 9], [29, 8]], ...]
    di_dgl_g = embed_traj(graph, agent_pos, total_tasks_bef, agent_traj)
    total_tasks_after = ag(di_dgl_g)
    total_tasks_bef = total_tasks_after

    costs = [len(t) for t in agent_traj]
    print("cost:{}".format(sum(costs)))

'''
EECBS 실행 parameter

m: the map file from the MAPF benchmark
a: the scenario file from the MAPF benchmark
o: the output file that contains the search statistics
outputPaths: the output file that contains the paths
k: the number of agents
t: the runtime limit
suboptimality: the suboptimality factor w
'''