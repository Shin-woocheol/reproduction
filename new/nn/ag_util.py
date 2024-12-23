import dgl
import networkx as nx
import torch
import time

from utils.astar import graph_astar

AG_type = 1
TASK_type = 2

def convert_to_bipartite(nx_g, agent_pos, task_pos, task_finished=[], astar_dist=None):
    '''
    nx_g : grid graph
    agent_pos : agent pos list
    task_pos : task pos list
    task_finished : task finish flag
    astar_dist : agent to task a start distance list

    task -> agent로의 directed bipartite graph생성.
    node feature에는 pos를, 
    edge feature로는 A*, menhatten, obstacle proxy를 넣음. 

    return : dgl graph
    '''
    node_idx_dict = {} #* grid graph에서의 좌표 저장.
    for i, node in enumerate(nx_g.nodes()): 
        node_idx_dict[tuple(node)] = i

    ag_node_indices = [] #* grid graph에서의 node index를 저장.
    for a in agent_pos:
        ag_node_indices.append(node_idx_dict[tuple(a)])

    task_node_indices = [] #* grid graph에서의 edge index를 저장.
    for task in task_pos: 
        task_node_indices.append(node_idx_dict[tuple(task)])

    node_indices = ag_node_indices + task_node_indices #* agent, task index를 이어붙임.
    all_locs = torch.tensor(list(nx_g.nodes())) #* graph내의 모든 node 좌표 list로 변환후 tensor로 변환.
    norm_locs = all_locs / 31 #* graph generate를 할 때도 모든 좌표를 size-1로 norm해줫음. 고로 32가 아니라 31이 맞을듯.

    n_ag = len(agent_pos) 
    n_task = len(task_pos) 

    # bipartite graph에서 사용할 edge 생성. dgl graph에서는 undirected를 지원하지 않음. 
    # 새롭게 생성될 graph에서 n_ag to n_ag + n_task -1 까지의 node는 task idx로 사용됨.
    src_idx = list(range(n_ag, n_ag + n_task)) * n_ag 
    dst_idx = [i for i in range(n_ag) for _ in range(n_task)] #각 task가 모든 agent로 이어지도록 dst 설정.
    bipartite_g = dgl.graph((src_idx, dst_idx))

    ag_type = torch.tensor([AG_type] * n_ag) 
    task_type = torch.tensor(task_finished) + 2 
    bipartite_g.ndata['type'] = torch.cat([ag_type, task_type], -1) #* type feature를 통해 agent라면 1, task끝나지 않았으면 2, 끝났으면 3임.
    #TODO 'loc' -> pos로 바꿈.
    bipartite_g.ndata['pos'] = norm_locs[node_indices] #* 각 agent, task node feature로 position넣어줌.
    bipartite_g.ndata['original_pos'] = all_locs[node_indices]
    #TODO dist -> astar로 바꿈, dist_m -> man_dist
    #* 아래 계산이 맞는지 확인 해야함.
    # astar_dist = astar_dist -1 #찍어보면 obs proxy 최솟값이 1임.
    bipartite_g.edata['astar_dist'] = torch.from_numpy(astar_dist.T.reshape(-1)) / 31 #*이것도 31일듯 근데 최대 거리는 62일텐데..
    bipartite_g.apply_edges(lambda edges: {'man_dist': (abs(edges.src['pos'] - edges.dst['pos'])).sum(-1)}) #* 아 이렇게 넣으면 최대 2가 되서 맞네.
    bipartite_g.edata['obs_proxy'] = bipartite_g.edata['astar_dist'] - bipartite_g.edata['man_dist']

    return bipartite_g
