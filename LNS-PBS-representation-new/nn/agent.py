import itertools
import dgl
import torch
import torch.nn as nn

from nn.gnn import GNN, Bipartite
from nn.memory import ReplayMemory

#* REINFORCE
class Agent(nn.Module):
    def __init__(self, embedding_dim=128, memory_size=50000, batch_size=100, gnn_layers=1):
        super(Agent, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Linear(2, embedding_dim) #? 왜 input dim이 2지? node embedding을 받는게 아니었나? -> 좌표를 받기 때문.
        self.gnn = GNN(in_dim=embedding_dim, out_dim=embedding_dim, embedding_dim=embedding_dim, n_layers=gnn_layers,
                       residual=True, ef_dim=3) #* ef dim 3인게, A*, menhatten, obstacle proxy 3개여서 그런듯.
        self.bipartite_policy = Bipartite(embedding_dim)

        self.replay_memory = ReplayMemory(capacity=memory_size, batch_size=batch_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.losses = []

    def forward(self, g, ag_order, continuing_ag, joint_action_prev, sample=True):
        # bs = g.batch_size
        n_ag = len(ag_order)
        policy = self.get_policy(g) #* 각 agent는 각 task에 대한 score를 가지고 있음.
        # print(f"policy : {policy}\n shape : {policy.shape}") torch.Size([20, 51]
        # print(f"policy : {policy[0]}")
        policy_temp = policy.clone().reshape(n_ag, -1)
        out_action = []
        for itr in range(n_ag):
            policy_temp[:, -1] = 1e-5  # dummy node 점수 보정
            agent_idx = ag_order[itr]
            # TODO: normalize prob?

            selected_ag_policy = policy_temp[agent_idx]
            if sample: #* 확률로 sampling할지, max score를 뽑을지.
                action = torch.distributions.Categorical(selected_ag_policy).sample()
            else:
                action = selected_ag_policy.argmax(-1)

            # mask continuing ag
            action[bool(continuing_ag[agent_idx])] = joint_action_prev[agent_idx].item()
            out_action.append(action.item())
            policy_temp[:, action] = 0 #* 선택한 action은 score 0으로 만듬. 이렇게 해서 priority가 되네.

        return out_action

    def forward_heuristic(self, g, ag_order, continuing_ag, joint_action_prev, sample=False, **kwargs):
        n_ag = len(ag_order)
        task_type = g.ndata['type'][n_ag:] == 2
        dists = g.edata['dist'].reshape(-1, n_ag).T + 1e-5
        dists[:, ~task_type] = 999

        # policy_temp = torch.softmax(1 / dists, -1)
        out_action = []
        for itr in range(n_ag):
            dists[:, -1] = 5  # dummy node
            agent_idx = ag_order[itr]
            selected_ag_dists = dists[agent_idx]

            action = selected_ag_dists.argmin(-1)

            # mask continuing ag
            action[bool(continuing_ag[agent_idx])] = joint_action_prev[agent_idx].item()
            out_action.append(action.item())
            dists[:, action] = 999

        return out_action

    def get_policy(self, g):
        nf, ef = self.generate_feature(g)  # one-hot encoded feature 'type'
        nf = self.embedding(nf) #* normalized된 좌표를 받아서 embedding_dim으로 embed
        out_nf = self.gnn(g, nf, ef) #* agent node에 대해서 node embedding이 담긴 tensor. (전체 node x embedding_dim)이지만 agent node것만 update되어있음.
        policy = self.bipartite_policy(g, out_nf) #* 각 node에 대해서 task에 대한 softmax score 반환 #? bipartite graph에서 task -> node edge만 존재하는지 확인해야함.
        policy[:, -1] = 1e-5 #* dummy task에 대한 score 보정 #? dummy가 왜 필요한거지? 선택하지 못하게 하는 것 같은데

        return policy

    def generate_feature(self, g):
        '''
        g : task- agent bipartite graph
        nf : normalized 좌표
        ef : A*, menhatten, obstacle proxy stack한 tensor
        '''
        # feature = torch.eye(3)[g.ndata['type']]
        # feature = torch.cat([feature, g.ndata['loc']], -1)
        nf = g.ndata['loc']
        if 'dist' not in g.edata.keys():
            g.apply_edges(lambda edges: {'dist': (abs(edges.src['loc'] - edges.dst['loc'])).sum(-1)})

        ef = torch.stack([g.edata['dist'], g.edata['dist_m'], g.edata['delay']], -1)

        return nf, ef

    def fit(self, baseline=0):
        gs, joint_action, ag_order, task_finished, next_t, terminated = self.replay_memory.episode_sample()
        bs = len(gs)
        gs = dgl.batch(gs)

        joint_action = torch.tensor(joint_action)
        all_action = joint_action.reshape(-1, 1)

        next_t = torch.tensor(next_t)

        policy = self.get_policy(gs)  # shape = bs * M, N
        _pol = policy.gather(-1, all_action)
        _pol = _pol.log()
        _pol[all_action == 20] = 0
        _pol = _pol.reshape(bs, -1) + 1e-4

        _logit = ((next_t - baseline).unsqueeze(-1) * _pol).mean(-1)
        loss = _logit.mean()
        # _logit = (next_t - baseline).sum(-1) * _pol.sum()
        # loss = _logit  # .mean()
        self.losses.append(loss)

        # behaved_agents = all_action < 20
        # selected_ag_pol = _pol[behaved_agents]

        # logit_sum = (selected_ag_pol + 1e-5).log().mean()
        # cost = next_t.sum()
        # loss = (cost - baseline) * (logit_sum)

        # TODO better loss design
        if len(self.losses) > 20:
            loss = torch.stack(self.losses).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            self.losses = []

        self.replay_memory.memory = []  # on-policy

        return {'loss': loss.item()}

    def push(self, *args):
        self.replay_memory.push([*args])
