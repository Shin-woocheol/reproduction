import subprocess
import numpy as np
from utils.astar import graph_astar
import time


def save_map(grid, filename):
    #* EECBS를 위한 map저장.
    f = open('EECBS/{}.map'.format(filename), 'w')
    f.write('type four-directional\n')
    f.write('height {}\n'.format(grid.shape[0]))
    f.write('width {}\n'.format(grid.shape[1]))
    f.write('map\n')

    # creating map from grid
    map_dict = {0: '.', 1: '@'}
    for r in range(grid.shape[0]):
        line = grid[r]
        l = []
        for g in line:
            l.append(map_dict[g])
        f.write(''.join(l) + '\n')

    f.close()

    f = open('DecAstar/{}.map'.format(filename), 'w')
    f.write('type four-directional\n')
    f.write('height {}\n'.format(grid.shape[0]))
    f.write('width {}\n'.format(grid.shape[1]))
    f.write('map\n')

    # # creating map from grid
    # map_dict = {0: '.', 1: '@'}
    # for r in range(grid.shape[0]):
    #     line = grid[r]
    #     l = []
    #     for g in line:
    #         l.append(map_dict[g])
    #     f.write(''.join(l) + '\n')

    f.close()


def save_scenario(agent_pos, total_tasks, scenario_name, row, column):
    #* 나중에 어떻게 쓰이는 것인지는 잘 모르겟지만, 전체 agent list, task list로 일단 맞춰서 세움.
    #* 일단 index맞는 task로 assign을 했다고 생각하는 것 같음. 그리고 나서 둘 사이의 좌표를 빼서 manhatten dist구함.
    #* 그래서 .scen에 적히는 내용은 grid size, agent 좌표 반대로, task 좌표 반대로, dist 이렇게 적힘.
    #* 나중에 뭐에 쓰기 위해서 적는 것인지는 모르겠음. 근데 EECBS repo에 이렇게 된 파일이 존재함.
    f = open('EECBS/{}.scen'.format(scenario_name), 'w')
    f.write('version 1\n')
    # print(f"agent_pos : {agent_pos} \n total tasks : {total_tasks}")
    for a, t in zip(agent_pos, total_tasks):
        # print(f"a : {a}, t : {t}\n")
        # t[0] : task pos list 좌표.
        task = t[0]  # TODO:add task seq 
        dist = abs(np.array(a) - np.array(t)).sum()  # Manhatten dist
        line = '1 \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}'.format('{}.map'.format(scenario_name), row, column, a[1],
                                                                  a[0], task[1], task[0], dist)
        f.write(line + "\n")
    f.close()


def save_scenario_dec(agent_pos, total_tasks, scenario_name, row=32, column=32):
    f = open('DecAstar/{}.scen'.format(scenario_name), 'w')
    f.write('version 1\n')
    for a, t in zip(agent_pos, total_tasks):
        task = t[0]  # TODO:add task seq
        dist = abs(np.array(a) - np.array(t)).sum()  # Manhatten dist
        line = '1 \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}'.format('{}.map'.format(scenario_name), row, column, a[1],
                                                                  a[0], task[1], task[0], dist)
        f.write(line + "\n")
    f.close()


def read_trajectory(path_file_dir):
    f = open(path_file_dir, 'r')
    lines = f.readlines() #각 줄을 list로 lines에 저장.
    agent_traj = []

    for i, string in enumerate(lines): #enumerate로 index와 value를 동시에 얻음.
        curr_agent_traj = []
        splitted_string = string.split('->') #['Agent 0: (0,8)', '(1,8)', '(2,8)', '(3,8)']
        for itr, s in enumerate(splitted_string):
            if itr == len(splitted_string) - 1:
                continue
            if itr == 0: #첫 번쨰 특별처리.
                tup = s.split(' ')[-1]
            else:
                tup = s
            #궤적 좌표를 받앗음.
            ag_loc = [int(i) for i in tup[1:-1].split(',')] #좌표 괄호 때고 숫자만 저장 list에.
            curr_agent_traj.append(ag_loc)
        agent_traj.append(curr_agent_traj)

    f.close()

    return agent_traj


def compute_astar(agent_pos, total_tasks, graph):
    dec_solver_path = "DecAstar/"
    n_ag = len(agent_pos)
    n_task = len(total_tasks)

    dists = []
    for task in total_tasks: #* 모든 task부터 모든 agent까지의 거리 astar로 잼.
        temp = []
        for ag in agent_pos:
            _, dist = graph_astar(graph, ag, task[0])
            temp.append(dist)
        dists.append(temp) #* task에 대해서 모든 agent까지의 거리를 append

    return np.array(dists) + 1 #? 왜 모든 거리에 +1을 수행하는거지?

    # process_out_dec = subprocess.run(dec_c, capture_output=True)
    # text_byte_dec = process_out_dec.stdout.decode('utf-8')
    # return text_byte_dec
