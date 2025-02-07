import dgl
import torch
import torch.nn as nn
from math import inf

AG_type = 1
TASK_type = 2
FIN_TASK_type = 3

#TODO over smoothing문제로 인해 mesage passing은 한번만 진행하려고 함. 그래서 아래 GNN, GNNlayer를 하나로 합쳐줄 수 있음.

class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim, ef_dim, n_layers, residual=False):
        #* agent에서 in_dim, out_dim, embedding_dim은 embedding dim으로, n_layer는 1로, ef_dim은 3으로 옴.
        super(GNN, self).__init__()
        _ins = [in_dim] + [embedding_dim] * (n_layers - 1)
        _outs = [embedding_dim] * (n_layers - 1) + [out_dim]
        layers = []
        for _i, _o in zip(_ins, _outs): 
            layers.append(GNNLayer(_i, _o, ef_dim, embedding_dim))
        self.layers = nn.ModuleList(layers)
        self.residual = residual

    def forward(self, g, nf, ef):
        nf_prev = nf
        for layer in self.layers: 
            nf = layer(g, nf_prev, ef)
            if self.residual:
                nf_prev = nf + nf_prev
            else:
                nf_prev = nf
        return nf #* agent node에 대해서 update된 node embedding을 반환.

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ef_dim, embedding_dim):
        super(GNNLayer, self).__init__()
        #* option1
        # self.node_embedding = nn.Sequential(nn.Linear(out_dim + in_dim, out_dim, bias=False),
        #                                     nn.LeakyReLU())
        # self.edge_embedding = nn.Sequential(nn.Linear(in_dim * 2 + ef_dim, out_dim, bias=False),
        #                                     nn.LeakyReLU())
        #* option2. 모든 neural net은 MLP with two layers of 128 unit
        # self.node_embedding = nn.Sequential(nn.Linear(out_dim + in_dim, embedding_dim, bias=False),
        #                                     nn.LeakyReLU(),
        #                                     nn.Linear(embedding_dim, out_dim, bias=False),
        #                                     nn.LeakyReLU())
        # self.edge_embedding = nn.Sequential(nn.Linear(in_dim * 2 + ef_dim, embedding_dim, bias=False),
        #                                     nn.LeakyReLU(),
        #                                     nn.Linear(embedding_dim, out_dim, bias=False),
        #                                     nn.LeakyReLU())
        #* option3. -표현에 제약을 주지 않기
        self.node_embedding = nn.Sequential(nn.Linear(out_dim + in_dim, embedding_dim, bias=False),
                                            nn.LeakyReLU(),
                                            nn.Linear(embedding_dim, out_dim, bias=False))
        self.edge_embedding = nn.Sequential(nn.Linear(in_dim * 2 + ef_dim, embedding_dim, bias=False),
                                            nn.LeakyReLU(),
                                            nn.Linear(embedding_dim, out_dim, bias=False))

    def forward(self, g: dgl.DGLGraph, nf, ef):
        '''
        dgl이 현재 task -> agent의 directed graph로 만들어졌으므로, 반대 edge추가로 undirected로 만든 후
        message passing으로 node embedding update.
        이게 논문 내용인듯.
        '''
        g_copy = g.clone()
        g_copy.ndata['nf'] = nf
        g_copy.edata['ef'] = ef

        #* option. 반대 edge 생성 후 message passing.
        #* 하면 task 또한 agent와 같은 방법으로 node embedding 생성.
        #* 안하면 task는 init embedding만 이용해서 다시 embed.
        ###
        src, dst = g_copy.edges()
        num_edges = len(src)
        g_copy.add_edges(dst, src) #reverse edge추가
        g_copy.edata['ef'][num_edges:] = g_copy.edata['ef'][:num_edges] #feature복사
        ###
        #message passing
        g_copy.update_all(message_func=self.message_func,
               reduce_func=self.reduce_func,
               apply_node_func=self.apply_node_func)
        
        out_nf = g_copy.ndata.pop('out_nf')
        return out_nf

    def message_func(self, edges): #* edge embedding
        ef = torch.concat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], -1)
        msg = self.edge_embedding(ef)
        return {'msg': msg}

    def reduce_func(self, nodes): #* summation으로 message aggregate.
        msg = nodes.mailbox['msg'].sum(1) 
        return {'red_msg': msg}

    def apply_node_func(self, nodes): #* 각 node에 대해서 현재 node feature와 aggregated message로 node_embedding.
        in_feat = torch.concat([nodes.data['nf'], nodes.data['red_msg']], -1)
        out_feat = self.node_embedding(in_feat) 
        return {'out_nf': out_feat} #agent node의 out_nf에 저장.

class Bipartite(nn.Module): #* GNN을 통해서 agent node embedding을 얻고 나서 bipartite graph와 함께 이 model 태움.
    def __init__(self, embedding_dim):
        super(Bipartite, self).__init__()
        #* two layers of 128 units each and LeakyRelu
        # self.score_layer = nn.Sequential( #* using pair of node embedding.
        #     nn.Linear(2 * embedding_dim, embedding_dim, bias=False),
        #     nn.BatchNorm1d(embedding_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(embedding_dim, 1, bias=False),
        #     nn.BatchNorm1d(1),
        #     nn.LeakyReLU()
        # )
        #* -표현 제한 x
        self.score_layer = nn.Sequential( #* using pair of node embedding.
            nn.Linear(2 * embedding_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1, bias=False),
            nn.BatchNorm1d(1)
        )

    def forward(self, g: dgl.DGLGraph, nf): #* nf는 GNN을 거쳐서 생성된 node embedding.
        '''
        nf : message passing을 통해 update된 node embedding.
        추가적인 layer를 통해서 각 task assign에 대한 score를 받고 그걸 softmax해서 policy로 반환해줌.
        '''
        g.ndata['nf'] = nf
        ag_node_indices = g.filter_nodes(ag_node_func) #* node type 1 인 것을 agent로 설정했음.
        g.ndata['finished'] = g.ndata['type'] == FIN_TASK_type #* fin task는 3으로 assign했음. task 끝난 것만 true. g만드는 과정 중에 설정해줌.
        g.update_all(message_func=self.message, reduce_func=self.reduce, apply_node_func=self.apply_node) #* 이전 push와는 달리 모든 node에 대해서 update.
        #* message를 통해서 오는 것이 score. 그리고 reduce과정에서 softmax를 취해줌.
        policy = g.ndata.pop('policy')[ag_node_indices]
        return policy

    def message(self, edges):
        src = edges.src['nf']  
        dst = edges.dst['nf'] #edge 수 x embedding dim
        m = torch.cat([src, dst], dim=1) 

        score = self.score_layer(m) #* 이 layer를 통해서 각 agent는 각 task에 대한 score 예측을 받음.

        task_finished = edges.src['finished']
        score[task_finished.bool()] = -inf #* finished task라면 -inf score.
        return {'score': score} #* 즉, 모든 노드는 모든 task에 대해서 score를 받음.

    def reduce(self, nodes):
        score = nodes.mailbox['score']
        score = score.squeeze()
        return {'policy': score}


    def apply_node(self, nodes):
        return {'policy': nodes.data['policy']} #* 각 node의 policy라는 feature에 softmax score넣음.


def ag_node_func(nodes):
    return nodes.data['type'] == AG_type  # .squeeze(1)


def task_node_func(nodes):
    return nodes.data['type'] == TASK_type  # .squeeze(1)


def all_task_node_func(nodes):
    return nodes.data['type'] == TASK_type or nodes.data['type'] == FIN_TASK_type  # .squeeze(1)
