import os
import os.path
import pickle
import random
from pathlib import Path

import numpy as np

from graph.generate_graph import gen_graph

curr_path = os.path.realpath(__file__)
scenario_dir = os.path.join(Path(curr_path).parent.parent, 'scenarios')

"""
1. Create random grid graph (user defined size, obstacle ratio)
2. Initialize the predefined positions of the agents and tasks
3. Save grid, graph, initial agent positions, task
"""



def save_scenarios(itr=1, size=32, obs=20, T=1, M=10, N=10, name=''):
    """
    T: task length -> if 2, tau=(s, g) #? 왜 필요하지? task별 특정 위치들을 방문해야 완료되도록 만들기 위함인듯.
    M: the number of agents
    N: the number of tasks
    """
    # instance, graph = gen_graph(size, obs)
    # vis_graph(graph)

    for it in range(itr):
        instance, graph = gen_graph(size, obs) 
        #* instance는 2차원 배열에 obstacle위치 1로 표시된 것, graph는 grid graph에서 obstacle은 끊음.
        #* 그리고 node에는 좌표, edge에는 distance feature가 들어가있음.

        empty_idx = list(range(len(graph))) #* graph node idx
        agent_idx = random.sample(empty_idx, M) #* 그중 random M개 선택. agent의 위치.
        agent_pos = np.array([a for a in graph])[agent_idx] #* grid graph여서 np.array로 좌표 tuple 모음이 나옴. 그래서 agent좌표를 뽑음.
        empty_idx = list(set(empty_idx) - set(agent_idx))

        # task 좌표를 np.array 형태로 저장
        task_idx = random.sample(empty_idx, N)  # task를 위해 N개 random 선택
        task_pos = np.array([t for t in graph])[task_idx]  # task 좌표 저장

        datas = [instance, graph, agent_pos, task_pos] #* 전체 map, available 길 graph, agent pos set, task pos list 
        dir = scenario_dir + '/{}{}{}_{}_{}_{}'.format(size, size, obs, T, M, N) + name + '/'

        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print("Error: Cannot create the directory.")

        with open(dir + 'scenario_{}.pkl'.format(it + 1), 'wb') as f:
            for d in datas:
                pickle.dump(d, f)

        with open(dir + 'scenario_{}.txt'.format(it + 1), 'w') as f:
            f.write("Instance:\n")
            f.write(np.array2string(instance, threshold=instance.size, max_line_width=10**9))
            f.write("\n\nAgent Positions:\n")
            f.write(str(agent_pos))
            f.write("\n\nTasks:\n")
            f.write(str(task_pos))
            f.write("\n\nGraph Nodes:\n")
            f.write(str(list(graph.nodes(data=True))))
            f.write("\n\nGraph Edges:\n")
            f.write(str(list(graph.edges(data=True))))
        f.close()


def load_scenarios(dir):
    dir = scenario_dir + '/' + dir
    data_list = []
    with open(dir, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
            except EOFError:
                break
            data_list.append(data)

    return data_list

if __name__ == "__main__":
    save_scenarios(itr=10, size=32, obs=20, T=1, M=10, N=20, name='_eval')
    save_scenarios(itr=20, size=32, obs=20, T=1, M=10, N=20, name='')
    
