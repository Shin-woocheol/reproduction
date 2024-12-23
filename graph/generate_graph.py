from copy import deepcopy

import networkx as nx
import numpy as np


# def gen_graph(size=32, obs=20, rand_coord=False):
#     instance = np.zeros((size, size)) #* square grid scenario
#     obstacle = np.random.random((size, size)) <= obs/100 #* size x size의 각 배열에 [0,1] assign 후, 저 수보다 작거나 같으면 true or false
#     instance[obstacle] = 1
#     g = tool(instance, rand_coord=rand_coord)
#     components = [c for c in nx.connected_components(g)] #* obstacle을 배치해도 모든 위치에 접근 가능하도록.

#     while len(components) != 1:
#         instance = np.zeros((size, size))
#         obstacle = np.random.random((size, size)) <= obs / 100
#         instance[obstacle] = 1
#         g = tool(instance, rand_coord=rand_coord)
#         components = [c for c in nx.connected_components(g)]

#     return instance, g

def gen_graph(size=32, obs=20, rand_coord=False):
    instance = np.zeros((size, size)) #* square grid scenario
    # obstacle = np.random.random((size, size)) <= obs/100 #* size x size의 각 배열에 [0,1] assign 후, 저 수보다 작거나 같으면 true or false
    # instance[obstacle] = 1
    #! obs density만큼 obstacle을 설치하도록 변경.
    num_obs = int(size * size / obs) #* obs density만큼 생성.
    obs_index = np.random.choice(np.arange(size*size), num_obs, replace=False)
    instance.ravel()[obs_index] = 1
    g = tool(instance, rand_coord=rand_coord)
    components = [c for c in nx.connected_components(g)] #* obstacle을 배치해도 모든 위치에 접근 가능하도록.

    while len(components) != 1:
        instance = np.zeros((size, size))
        num_obs = int(size * size / obs) #* obs density만큼 생성.
        obs_index = np.random.choice(np.arange(size*size), num_obs, replace=False)
        instance.ravel()[obs_index] = 1
        # obstacle = np.random.random((size, size)) <= obs / 100
        # instance[obstacle] = 1
        g = tool(instance, rand_coord=rand_coord)
        components = [c for c in nx.connected_components(g)]

    return instance, g

def tool(instance, rand_coord=False):
    '''
    만들어진 grid graph의 node에 0-1 normalized된 좌표를 loc feature로 넘김.
    obstacle이 있는 node는 graph에서 제거
    edge에 dist feature로는 node사이의 loc feature의 euclidian distance 넘김.
    '''
    instance = deepcopy(instance)
    m, n = instance.shape[0], instance.shape[1]
    g = nx.grid_2d_graph(m, n)

    if rand_coord:
        rand_interval_x = np.random.random(m - 1) + .5
        rand_x = np.array([rand_interval_x[:i].sum() for i in range(m)])
        rand_x /= rand_x.sum()

        rand_interval_y = np.random.random(n - 1) + .5
        rand_y = np.array([rand_interval_y[:i].sum() for i in range(n)])
        rand_y /= rand_y.sum()

    else: #* 보니까 좌표를 normalize해서 저장하고 있음. 나중에 menhaten같은 경우는 step을 세서 하려나? 
        rand_x = np.array([0 + i / (n - 1) for i in range(m)])
        rand_y = np.array([0 + i / (m - 1) for i in range(n)])

    xs = np.array(list(rand_x) * n)
    ys = rand_y[::-1].repeat(m)
    coords = np.stack([xs, ys], -1)

    for id, n_id in enumerate(g.nodes()): #* 각 node에 좌표를 loc 이라는 feature로 넣어줌.
        g.nodes[n_id]['loc'] = coords[id].tolist()
        # g.nodes[n_id]['type'] = int(instance.reshape(-1)[id])

    for r, c in zip(instance.nonzero()[0], instance.nonzero()[1]): #* obstacle 제거
        g.remove_node((r, c))

    #* e_id = (i, j)
    #* 음.. 지금 유클리디안으로 거리를 계산해주고 있는데, 어차피 위 아래 연결 grid그래프여서 상관은 없을듯.
    #* 그리고 normalized된 좌표를 가지도록 설정한 것이 network에서 예측같은거 할때 반환하기 좋게 하려고?
    for id, e_id in enumerate(g.edges()):
        loc = (np.array(g.nodes[e_id[0]]['loc']) - np.array(g.nodes[e_id[1]]['loc'])) ** 2
        dist = loc.sum(-1).reshape(-1, 1) ** .5
        g.edges[e_id]['dist'] = dist

    return g

if __name__ == '__main__':
    instance, g = gen_graph()
    
    with open("output.txt", "w") as f:
        f.write("gen_graph test\n")
        f.write("instance:\n")
        f.write(np.array2string(instance, threshold=instance.size, max_line_width=10**9)) 
        f.write("\n\n")
        f.write("graph:\n")
        f.write(str(g))
        f.write("\n")
    
    print("test result : output.txt")