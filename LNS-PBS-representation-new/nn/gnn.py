import dgl
import torch
import torch.nn as nn
from math import inf

AG_type = 1
TASK_type = 2
FIN_TASK_type = 3


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim, ef_dim, n_layers, residual=False):
        #* agent에서 in_dim, out_dim, embedding_dim은 embedding dim으로, n_layer는 1로, ef_dim은 3으로 옴.
        super(GNN, self).__init__()
        _ins = [in_dim] + [embedding_dim] * (n_layers - 1)
        _outs = [embedding_dim] * (n_layers - 1) + [out_dim]

        layers = []
        for _i, _o in zip(_ins, _outs): 
            layers.append(GNNLayer(_i, _o, ef_dim))
        self.layers = nn.ModuleList(layers)

        self.residual = residual

    def forward(self, g, nf, ef):
        nf_prev = nf
        for layer in self.layers:
            nf = layer(g, nf_prev, ef) #* bipartite graph, node feature, edge feature같이 넘김.
            if self.residual:
                nf_prev = nf + nf_prev
            else:
                nf_prev = nf
        return nf #* agent node에 대해서 update된 node embedding을 반환.

#! 아래 GNNLayer에서는 task to agent로의 message만을 정의함.
#! 그렇다는 것은, GNN layer를 2개 이상으로 해도 같은 message를 받는 것 밖에 되지 않음.
class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ef_dim):
        super(GNNLayer, self).__init__()
        self.node_embedding = nn.Sequential(nn.Linear(out_dim + in_dim, out_dim, bias=False),
                                            nn.LeakyReLU())
        self.edge_embedding = nn.Sequential(nn.Linear(in_dim * 2 + ef_dim, out_dim, bias=False),
                                            nn.LeakyReLU())

    def forward(self, g: dgl.DGLGraph, nf, ef):
        g.ndata['nf'] = nf
        g.edata['ef'] = ef
        task_node_idx = g.filter_nodes(task_node_func) #* task type에 해당하는 노드 필터링. 2니까 task면서 complete되지 않은거.
        g.push(u=task_node_idx,
               message_func=self.message_func,
               reduce_func=self.reduce_func,
               apply_node_func=self.apply_node_func) #* message passing.
        """
        u : message passing 시작 노드 집합
        message_func : 각 edge에서의 message 생성 정의
        reduce_func : target node에서 받은 message aggregation 방법 정의
        apply_node_func : message를 이용해서 node feature update방법 정의
        
        task node가 연결된 엣지를 따라 message 생성 후 target node로 전달.
        agent node에 대해서만 apply_node_func를 통해 embedding 됨.
        """

        out_nf = g.ndata.pop('out_nf') #* update된 node feature pop
        # print(f"out_nf : {out_nf}\n shape : {out_nf.shape}") torch.Size([71, 128]) agent 20 + task 50 + dummy 1 그리고 agent만 update됨.
        g.ndata.pop('nf')
        g.edata.pop('ef')
        return out_nf

    def message_func(self, edges):
        ef = torch.concat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], -1) #* 두 node의 feature와 edge feature를 concat
        msg = self.edge_embedding(ef) #* 그렇기 때문에 위에 정의된 edge_embedding에서 받는 것이 imputdim의 2배 + edge feature인 것.
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
        self.embedding_dim = embedding_dim
        self.attention_fc = nn.Sequential(nn.Linear(2 * embedding_dim, 1, bias=False),
                                          nn.LeakyReLU()
                                          ) #* 두 node에 대한 embedidng을 받아서.

        self.ag_score = nn.Sequential(nn.Linear(embedding_dim, 1, bias=False), nn.LeakyReLU())

        # Todo:transformer
        # self.K = nn.Linear(embedding_dim, embedding_dim)
        # self.Q = nn.Linear(embedding_dim, embedding_dim)
        # self.V = nn.Linear(embedding_dim, embedding_dim)

    """
    Assume ag_size and task_size does not vary within batch
    """

    def forward(self, g: dgl.DGLGraph, nf): #* nf는 GNN을 거쳐서 생성된 node embedding.
        '''
        추가적인 layer를 통해서 각 task assign에 대한 score를 받고 그걸 softmax해서 policy로 반환해줌.
        '''
        g.ndata['nf'] = nf

        ag_node_indices = g.filter_nodes(ag_node_func) #* agent node filtering.
        g.ndata['finished'] = g.ndata['type'] == FIN_TASK_type #* task 끝난 것만 true.
        g.update_all(message_func=self.message, reduce_func=self.reduce, apply_node_func=self.apply_node) #* 이전 push와는 달리 모든 node에 대해서 update.
        #* message를 통해서 오는 것이 score. 그리고 reduce과정에서 softmax를 취해줌.
        policy = g.ndata.pop('policy')[ag_node_indices]
        return policy

    def forward_prev(self, g: dgl.DGLGraph, bipartite_g: dgl.DGLGraph, nf, ag_node_indices, task_node_indices,
                     task_finished):
        n_batch = g.batch_size
        g.ndata['nf'] = nf

        ag_nfs = g.nodes[ag_node_indices].data['nf']
        task_nfs = g.nodes[task_node_indices].data['nf'][~task_finished]

        # pull from task node idx to agent node idx
        ag_node_indices = bipartite_g.filter_nodes(ag_node_func)
        task_node_indices = bipartite_g.filter_nodes(task_node_func)

        bipartite_g.ndata['finished'] = torch.ones(bipartite_g.number_of_nodes(), 1)
        bipartite_g.nodes[ag_node_indices].data['nf'] = ag_nfs
        bipartite_g.nodes[task_node_indices].data['nf'] = task_nfs
        bipartite_g.nodes[task_node_indices].data['finished'] = torch.zeros(task_nfs.shape[0], 1)

        bipartite_g.update_all(message_func=self.message, reduce_func=self.reduce,
                               apply_node_func=self.apply_node)

        policy = bipartite_g.ndata.pop('policy')[ag_node_indices]
        # policy = policy.reshape(n_batch, -1, policy.shape[-1])

        ag_score = self.ag_score(ag_nfs).squeeze().reshape(n_batch, -1)
        ag_policy = torch.softmax(ag_score, -1)

        return policy, ag_policy

    def message(self, edges): #? 여기 마치 bipartite graph에서 task -> agent로의 edge만 남아있는 것 같이 짰음. 확인해야함.
        src = edges.src['nf']
        dst = edges.dst['nf']
        m = torch.cat([src, dst], dim=1)
        score = self.attention_fc(m) #* 이 layer를 통해서 각 agent는 각 task에 대한 score 예측을 받음.
        task_finished = edges.src['finished']
        # score = score - inf * task_finished.bool()
        score[task_finished.bool()] = -inf #* finished task라면 -inf score.

        # Todo:self-attention
        # K = self.K(m)
        # Q = self.Q(nf)
        #
        # score = (K * Q).sum(-1) / self.embedding_dim  # shape = (ag, task)
        # policy = torch.softmax(A, -1)

        return {'score': score} #* 즉, 모든 노드는 모든 task에 대해서 score를 받음.

    def reduce(self, nodes):
        score = nodes.mailbox['score']
        policy = torch.softmax(score, 1).squeeze() #* 각 task에 대한 score를 softmax. 이게 각 agent가 각 task를 정할 확률인듯.
        return {'policy': policy}

    def apply_node(self, nodes):
        return {'policy': nodes.data['policy']} #* 각 node의 policy라는 feature에 softmax score넣음.


def ag_node_func(nodes):
    return nodes.data['type'] == AG_type  # .squeeze(1)


def task_node_func(nodes):
    return nodes.data['type'] == TASK_type  # .squeeze(1)


def all_task_node_func(nodes):
    return nodes.data['type'] == TASK_type or nodes.data['type'] == FIN_TASK_type  # .squeeze(1)
