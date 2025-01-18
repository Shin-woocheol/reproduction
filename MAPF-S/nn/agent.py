import itertools
import dgl
import torch
import torch.nn as nn
import numpy as np

from nn.gnn import GNN, Bipartite
from nn.memory import RolloutBuffer

#* REINFORCE
class Agent(nn.Module):
    def __init__(self, embedding_dim=128, gnn_layers=1, lr = 1e-5, gpu = False):
        super(Agent, self).__init__()
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() and gpu == True else "cpu")
        self.embedding_dim = embedding_dim
        #* option1
        # self.init_node_embedding = nn.Linear(2, embedding_dim) #* node feature -> init node embedding.
        #* option2
        self.init_node_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU()
        ).to(self.device)
        self.gnn = GNN(in_dim=embedding_dim, out_dim=embedding_dim, embedding_dim=embedding_dim, n_layers=gnn_layers,
                       residual=True, ef_dim=3).to(self.device) #* ef dim 3인게, A*, menhatten, obstacle proxy 3개
        self.bipartite_policy = Bipartite(embedding_dim).to(self.device)
        self.buffer = RolloutBuffer() 
        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.losses = []

    def forward(self, g, continuing_ag, joint_action_prev, train=True):
        g = g.to(self.device)

        policy = self.get_policy(g) 
        policy_temp = policy.clone()

        M, N = policy_temp.shape
        out_action = [-1] * M 
            
        # continuing agent 처리
        for i in range(M):
            if continuing_ag[i]:
                out_action[i] = joint_action_prev[i]
                policy_temp[i, :] = 0.0
                policy_temp[:, out_action[i]] = 0.0

        policy_flat = policy_temp.flatten()  # (M*N,)

        for i in range(M):
            print(f"policy : {policy_temp}\n")
            if torch.all(policy_flat == 0):
                break

            if continuing_ag[i]:
                continue

            if train:
                dist = torch.distributions.Categorical(policy_flat)
                chosen_idx = dist.sample().item()
                print(f"chosen_action : {policy_flat[chosen_idx]}\n")
            else:
                chosen_idx = policy_flat.argmax().item()

            agent_idx = chosen_idx // N
            task_idx  = chosen_idx % N

            out_action[agent_idx] = task_idx

            policy_temp[agent_idx, :] = 0.0
            policy_temp[:, task_idx]  = 0.0
            policy_flat = policy_temp.flatten()

        return out_action

    def get_policy(self, g):
        '''
        agent에 대해 각 task에 대한 score반환.
        '''
        nf, ef = self.generate_feature(g)
        nf = self.init_node_embedding(nf)
        out_nf = self.gnn(g, nf, ef)
        # 여기까지 오면 agent node의 feature는 task로 부터 message passing받았음. score layer에 넣기 직전 embedding
        # task에 대한 score반환. (M, N) shape
        score = self.bipartite_policy(g, out_nf)
        # Flatten (M, N) -> (M*N,)
        policy_1d = score.flatten() 
        softmax_1d = torch.softmax(policy_1d, dim=0)
        policy = softmax_1d.view(score.shape)  # shape: [10, 20]
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
            #* option2. continuing을 선택하는 것도 action으로 보는 것.
            num_tot_action = len(joint_action) * len(joint_action[0])

            gs = dgl.batch(gs).to(self.device)
            joint_action = torch.tensor(joint_action).to(self.device) #* 각 agent에 할당된 task. (iter x num_agent) ex)18 x 10
            all_action = joint_action.reshape(-1, 1)
            next_t = torch.tensor(next_t).to(self.device) #* 가장 빨리 끝나는 다음 task 종료까지의 step. (iter,)

            policy = self.get_policy(gs)  # shape = bs * M, N
            mask = (all_action != -1)
            valid_action = all_action[mask].reshape(-1, 1)
            valid_policy = policy[mask.squeeze()].gather(-1, valid_action)
            valid_policy = torch.clamp(valid_policy, min=1e-10) #* 안해주면 log시 -inf의 발생으로 weight가 nan으로 update됨.
            valid_policy_log = valid_policy.log()
            #* option1. continuing 제외 선택.
            # pol_sum = torch.mean(valid_policy_log) #* valid action의 수가 다르니까 mean을 하는게 맞는 것 같음.
            #* option2.
            pol_sum = torch.sum(valid_policy_log) / num_tot_action
            cost = torch.sum(next_t)

            # loss = -cost * pol_sum
            #* option 2
            loss = -(1/cost * pol_sum)

            total_loss += loss

        total_loss /= len(batch_traj)

        self.optimizer.zero_grad()
        total_loss.backward()
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                assert False, f"NaN detected in gradients of {name}"

        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        self.buffer.clear()
        return total_loss.item()
    
    def push(self, episode_traj):
        self.buffer.store(episode_traj)
