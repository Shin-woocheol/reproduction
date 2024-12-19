import itertools
import dgl
import torch
import torch.nn as nn
import numpy as np

from nn.gnn import GNN, Bipartite
from nn.memory import RolloutBuffer

#* REINFORCE
class Agent(nn.Module):
    def __init__(self, embedding_dim=128, gnn_layers=1, lr = 1e-5):
        super(Agent, self).__init__()
        self.embedding_dim = embedding_dim
        # self.init_node_embedding = nn.Linear(2, embedding_dim) #* node feature -> init node embedding.
        #*option2
        self.init_node_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU()
        )
        self.gnn = GNN(in_dim=embedding_dim, out_dim=embedding_dim, embedding_dim=embedding_dim, n_layers=gnn_layers,
                       residual=True, ef_dim=3) #* ef dim 3인게, A*, menhatten, obstacle proxy 3개
        self.bipartite_policy = Bipartite(embedding_dim)
        self.buffer = RolloutBuffer() 
        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.losses = []

    def forward(self, g, ag_order, continuing_ag, joint_action_prev, train=True):
        n_ag = len(ag_order)
        policy = self.get_policy(g) #* 각 agent는 각 task에 대한 score를 가지고 있음.
        print(f"policy : {policy}\n shape : {policy.shape}") #torch.Size([20, 51]
        policy_temp = policy.clone().reshape(n_ag, -1)
        out_action = []
        for itr in range(n_ag):
            agent_idx = ag_order[itr] #* agent order대로 action선택.
            selected_ag_policy = policy_temp[agent_idx]

            if torch.all(selected_ag_policy == 0): #* 이미 모든 task가 assign된 경우, -1로 처리.
                out_action.append(-1)
                continue

            if train:
                action = torch.distributions.Categorical(selected_ag_policy).sample().item()
            else:
                action = selected_ag_policy.argmax(-1).item()

            # mask continuing ag
            if continuing_ag[agent_idx]: #* 새롭게 rescheduling을 하지 않는 것이라면
                action = joint_action_prev[agent_idx].item() #* 이전에 schedule된 task idx
            out_action.append(action)
            policy_temp[:, action] = 0 #* 선택한 action은 score 0으로 만듬. 이렇게 해서 priority가 되네.

        return out_action

    def get_policy(self, g):
        '''
        agent에 대해 각 task에 대한 score반환.
        '''
        nf, ef = self.generate_feature(g)
        nf = self.init_node_embedding(nf)
        out_nf = self.gnn(g, nf, ef)
        # 여기까지 오면 agent node의 feature는 task로 부터 message passing받았음.
        policy = self.bipartite_policy(g, out_nf) #* 각 node에 대해서 task에 대한 softmax score 반환 #? bipartite graph에서 task -> node edge만 존재하는지 확인해야함.
        return policy

    def generate_feature(self, g):
        '''
        g : task- agent bipartite graph
        nf : normalized 좌표
        ef : A*, menhatten, obstacle proxy stack한 tensor
        '''
        nf = g.ndata['pos']
        #* option1. use obs_proxy
        ef = torch.stack([g.edata['astar_dist'], g.edata['man_dist'], g.edata['obs_proxy']], -1)
        #* option2. use astar and man only
        # ef = torch.stack([g.edata['astar_dist'], g.edata['man_dist']], -1)
        return nf, ef
    
    def learn(self, baseline=0):
        batch_traj = self.buffer.sample()
        total_loss = 0
        for traj in batch_traj:
            gs, joint_action, next_t = list(map(list, zip(*traj)))
            gs = dgl.batch(gs)
            joint_action = torch.tensor(joint_action) #* 각 agent에 할당된 task. (iter x num_agent) ex)18 x 10
            all_action = joint_action.reshape(-1, 1)
            next_t = torch.tensor(next_t) #* 가장 빨리 끝나는 다음 task 종료까지의 step. (iter,)

            policy = self.get_policy(gs)  # shape = bs * M, N
            mask = (all_action != -1)
            valid_action = all_action[mask].reshape(-1, 1)
            valid_policy = policy[mask.squeeze()].gather(-1, valid_action)
            valid_policy_log = valid_policy.log()
            pol_sum = torch.mean(valid_policy_log) #* valid action의 수가 다르니까 mean을 하는게 맞는 것 같음.
            cost = torch.sum(next_t)

            loss = cost * pol_sum

            total_loss += loss

        total_loss /= len(batch_traj)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        self.buffer.clear()
        return total_loss.item()

        gs, joint_action, ag_order, task_finished, next_t, terminated = self.buffer.sample() #* 한 episode에 대한 것.
        #* 즉, 각 요소마다 batch length 만큼의 요소가 잇을듯.

        gs = dgl.batch(gs) #* graph batch처리.
        #* 그냥 joint_action의 경우, list안에 list로 되어있음 len num_agent짜리.
        joint_action = torch.tensor(joint_action) #* 각 agent에 할당된 task. (iter x num_agent) ex)18 x 10
        all_action = joint_action.reshape(-1, 1) #* 행을 자동으로 하고 열을 1로 맞춤. (iter*num_agent x 1) ex) 180 x 1
        # 이제는 -1 index때문에 확률 처리가 안되네..?
        ###
        next_t = torch.tensor(next_t) #* 가장 빨리 끝나는 다음 task 종료까지의 step. (iter,)
        policy = self.get_policy(gs)  # shape = bs * M, N
        mask = (all_action != -1)
        valid_action = all_action[mask].reshape(-1, 1)
        valid_policy = policy[mask.squeeze()].gather(-1, valid_action)
        valid_policy_log = valid_policy.log()
        pol_sum = torch.mean(valid_policy_log) #* valid action의 수가 다르니까 mean을 하는게 맞는 것 같음.
        cost = torch.sum(next_t)
        #* option 1
        loss = -cost * pol_sum
        #* option 2
        # loss = -(1/cost * pol_sum)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    # def push(self, *args):
    #     self.buffer.store([*args])
    # [[],[],[]]
    def push(self, episode_traj):
        self.buffer.store(episode_traj)
