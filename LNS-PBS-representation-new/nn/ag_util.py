import dgl
import networkx as nx
import torch
import time

from utils.astar import graph_astar

AG_type = 1
TASK_type = 2


def process_graph(nx_g):
    dgl_g = dgl.from_networkx(nx_g, node_attrs=['loc', 'type'])
    # Manhatten dist; useless as all the edge dists are the same
    # dgl_g.apply_edges(lambda edges: {'dist_m': (abs(edges.src['loc'] - edges.dst['loc'])).sum(-1)})

    return dgl_g


def convert_dgl(nx_g, agent_pos, task_pos, task_finished=[], shortest_path=None):
    '''
    bipartite graph를 만들고, node feature, edge feature도 넣어줌. 논문에서의 그대로
    '''
    di_nx_g = nx.DiGraph(nx_g) #* directed graph로 만듬. undirected였던 edge는 두개의 directed + feature가 복사된다고 함.
    node_idx_dict = dict()

    for i, node in enumerate(di_nx_g.nodes()): #* graph의 각 노드 좌표를 key로 index 만듬.
        node_idx_dict[tuple(node)] = i
    # print(f"node_idx_dict : {node_idx_dict}")

    ag_node_indices = []
    for a in agent_pos:
        ag_node_indices.append(node_idx_dict[tuple(a)]) #*agent position을 받은 것을 이용해서 agent의 idx를 list에 저장
    # print(f"ag_node_indices : {ag_node_indices}")

    task_node_indices = []
    assert len(task_pos[0]) == 1, "task size > 1 not supported yet"
    for task in task_pos: #* 마찬가지로 task idx저장한 list만듬.
        task_node_indices.append(node_idx_dict[tuple(task[0])])
    # print(f"task_node_indices : {task_node_indices}")

    node_indices = ag_node_indices + task_node_indices #* agent, task pos 저장한 index.
    all_locs = torch.tensor(list(di_nx_g.nodes())) #* graph내의 모든 node 좌표 list로 변환후 tensor로 변환.
    #! size 32
    norm_locs = all_locs / 32 #* dist구할때도 normalize했으니까. size 32일 때.
    #? 근데 loc이 0부터 시작하니까 31로 나눠줘야 하는 것 아닌가.

    # option 1. Astar cost (느림)
    # edge_cost = []
    # for t in task_pos:
    #     for a in agent_pos:
    #         path_cost = graph_astar(nx_g, a, t[0])[1]
    #         edge_cost.append(path_cost)

    # option 2. Manhttan dist

    # generate bipartite graph; to be used in the policy network
    n_ag = len(agent_pos) #* agent 수
    n_task = len(task_pos) #* task 수

    src_idx = list(range(n_ag, n_ag + n_task)) * n_ag #task
    dst_idx = [i for i in range(n_ag) for _ in range(n_task)] #agent
    # result = []
    # for i in range(n_ag):
    #     for _ in range(n_task):
    #         result.append(i)
    bipartite_g = dgl.graph((src_idx, dst_idx)) #* complete bipartite graph 생성.
    # 위 함수의 동작이, 두 list의 idx가 맞게 tuple을 구성한 것을 edge로 사용하는 것.

    ag_type = torch.tensor([AG_type] * n_ag) # 1의 값을 가지는 elem이 n_ag개인 list.
    task_type = torch.tensor(task_finished) + 2 # false엿던 것은 2로, true 였던 것은 3으로 됨.

    #* bipartite graph에 node feature를 넣어줌.
    bipartite_g.ndata['type'] = torch.cat([ag_type, task_type], -1) # tensor 마지막 차원으로 concat
    bipartite_g.ndata['loc'] = norm_locs[node_indices] #agent, task에 대해 normalized된 location 반환
    bipartite_g.ndata['original_loc'] = all_locs[node_indices] #normalized안된 좌표.

    #* edge feature 저장.
    if shortest_path is None: #* A*계산한거 없으면, menhatten만 넣기
        bipartite_g.apply_edges(
            lambda edges: {'dist': (abs(edges.src['loc'] - edges.dst['loc'])).sum(-1)})
    else: #* A* 계산한 거 있으면, edge에 dist라는 속성으로는 A* 넣어주고, dist_m이라는 속성으로는 menhatten넣어줌.
        shortest_path = shortest_path - 1
        bipartite_g.edata['dist'] = torch.from_numpy(shortest_path.T.reshape(-1)) / 32 #! 여기도 size
        bipartite_g.apply_edges(
            lambda edges: {'dist_m': (abs(edges.src['loc'] - edges.dst['loc'])).sum(-1)})

        # gap between the shortest path and heuristic #* 이 부분이 논문에서 쓴 obstacle proxy
        bipartite_g.edata['delay'] = bipartite_g.edata['dist'] - bipartite_g.edata['dist_m']

    # add dummy task
    bipartite_g.add_edges([n_ag + n_task] * n_ag, range(n_ag)) #?모든 agent와 연결되는 한 task를 추가함 왜?

    return bipartite_g, ag_node_indices, task_node_indices


def convert_dgl_prev(nx_g, agent_pos, tasks, agent_traj, task_finished=[]):
    di_nx_g = nx.DiGraph(nx_g)  # default networkx graph is undirected. Change to the directed graph here
    # set default edge attribute
    nx.set_edge_attributes(di_nx_g, 0, 'traj')
    nx.set_node_attributes(di_nx_g, 0, 'type')

    # set visited traj edge attribute
    # Currently not using
    for t in agent_traj:
        for _f, _t in zip(t[:-1], t[1:]):
            # TODO: how to represent 'hold' action
            if _f == _t: continue
            di_nx_g.edges[tuple(_f), tuple(_t)]['traj'] = 1

    for p in agent_pos:
        di_nx_g.nodes[tuple(p)]['type'] = AG_type

    for finished, t in zip(task_finished, tasks):
        for _t in t:
            if not finished:
                di_nx_g.nodes[tuple(_t)]['type'] = TASK_type

    di_dgl_g = dgl.from_networkx(di_nx_g, node_attrs=['loc', 'type'], edge_attrs=['traj'])
    node_idx_dict = dict()
    coord_x = []
    coord_y = []
    for i, node in enumerate(di_nx_g.nodes()):
        node_idx_dict[tuple(node)] = i
        coord_x.append(node[0])
        coord_y.append(node[1])

    ag_node_indices = []
    for a in agent_pos:
        ag_node_indices.append(node_idx_dict[tuple(a)])

    task_node_indices = []
    assert len(tasks[0]) == 1, "task size > 1 not supported yet"
    for task in tasks:
        task_node_indices.append(node_idx_dict[tuple(task[0])])

    di_dgl_g.ndata['x'] = torch.tensor(coord_x)
    di_dgl_g.ndata['y'] = torch.tensor(coord_y)

    # generate bipartite graph; to be used in the policy network
    n_ag = len(agent_pos)
    n_task = len(tasks)  # - sum(task_finished)

    src_idx = list(range(n_ag, n_ag + n_task)) * n_ag
    dst_idx = [i for i in range(n_ag) for _ in range(n_task)]
    bipartite_g = dgl.graph((src_idx, dst_idx))

    ag_type = torch.tensor([AG_type] * n_ag)
    task_type = torch.tensor(task_finished) + 2

    bipartite_g.ndata['type'] = torch.cat([ag_type, task_type], -1)

    # assert (di_dgl_g.ndata['type'] == 1).sum() == n_ag

    return di_dgl_g, bipartite_g, ag_node_indices, task_node_indices


# def embed_traj(graph, agent_pos, total_tasks_bef, agent_traj):
#     di_dgl_g, _, _, _ = convert_dgl(graph, agent_pos, total_tasks_bef, agent_traj)
#     return di_dgl_g

def embed_traj(nx_g, agent_pos, tasks, agent_traj):
    di_nx_g = nx.DiGraph(nx_g)
    # set default edge attribute
    nx.set_edge_attributes(di_nx_g, 0, 'traj')
    nx.set_node_attributes(di_nx_g, 0, 'type')

    # set visited traj edge attribute
    for t in agent_traj:
        for _f, _t in zip(t[:-1], t[1:]):
            di_nx_g.edges[tuple(_f), tuple(_t)]['traj'] = 1

    for p in agent_pos:
        di_nx_g.nodes[tuple(p)]['type'] = AG_type

    for t in tasks:
        for _t in t:
            di_nx_g.nodes[tuple(_t)]['type'] = TASK_type

    di_dgl_g = dgl.from_networkx(di_nx_g, node_attrs=['loc', 'type'], edge_attrs=['traj'])
    return di_dgl_g

if __name__ == '__main__':
    from utils.generate_scenarios import load_scenarios

    M, N = 20, 50
    scenario = load_scenarios('323220_1_{}_{}/scenario_1.pkl'.format(M, N))
    grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]

    di_dgl_g = convert_dgl(graph, agent_pos, total_tasks, [])
