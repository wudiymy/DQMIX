import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.projection import proj
from utils.conv import bconv
import numpy as np

class DQMixer(nn.Module):
    def __init__(self, args):
        super(DQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim #32
        self.n_atom = args.n_atom
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.value_range = np.linspace(args.v_min, args.v_max, args.n_atom)
        self.value_range = th.FloatTensor(self.value_range)
        self.q_tot_value_range1 = np.linspace(args.n_agents * args.v_min, args.n_agents * args.v_max, 
                                             args.n_agents * args.n_atom - args.n_agents + 1)
        self.q_tot_value_range2 = np.linspace(self.embed_dim * args.v_min, self.embed_dim * args.v_max, 
                                             self.embed_dim * args.n_atom - self.embed_dim + 1)
        self.q_tot_value_range1 = th.FloatTensor(self.q_tot_value_range1)
        self.q_tot_value_range2 = th.FloatTensor(self.q_tot_value_range2)
        if args.use_cuda:
            self.value_range = self.value_range.cuda()
            self.q_tot_value_range1 = self.q_tot_value_range1.cuda()
            self.q_tot_value_range2 = self.q_tot_value_range2.cuda()

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed #64
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))


    def forward(self, agent_qs_distri, states):
        #agent_qs_distri: [bs, episode_len, n_agents, n_atom]
        #states.size(): [bs, episode_len, state_shape]
        bs = agent_qs_distri.size(0)
        episode_len = agent_qs_distri.size(1)
        n_agents = agent_qs_distri.size(2)
        states = states.reshape(-1, self.state_dim)                 #[bs*episode_len, state_shape]
        w1 = th.abs(self.hyper_w_1(states))                         #[bs*episode_len, n_agents*embed_dim]
        w1 = w1.view(bs, -1, self.embed_dim, self.n_agents, 1)      #[bs, episode_len, embed_dim, n_agents, 1]
        b1 = self.hyper_b_1(states)                                 #[bs*episode_len, embed_dim]
        b1 = b1.view(bs, -1, self.embed_dim, 1)                     #[bs, episode_len, embed_dim, 1]

        x = th.zeros((bs, episode_len, self.embed_dim, self.n_atom), device=agent_qs_distri.device)
        value_range = self.value_range.unsqueeze(0).unsqueeze(0).expand(bs, episode_len, -1) #[bs, episode_len, n_atom]
        for k in range(self.embed_dim):
            agent_qs_distri_w = th.zeros_like(agent_qs_distri, device=agent_qs_distri.device)
            for i in range(n_agents):
                next_v_range = value_range * w1[:, :, k, i] #[bs, episode_len, n_atom]
                agent_qs_distri_w[:, :, i] = proj(next_v_range, agent_qs_distri[:, :, i], self.v_min, self.v_max, self.n_atom)

            q_tot = agent_qs_distri_w[:, :, 0]
            for i in range(1, n_agents):
                q_tot = bconv(q_tot, agent_qs_distri_w[:, :, i])
            
            q_tot_value_range = self.q_tot_value_range1.unsqueeze(0).unsqueeze(0).expand(bs, episode_len, -1) #[bs, episode_len, N_ATOM]
            q_tot_value_range = q_tot_value_range + b1[:, :, k]
            #elu activte
            q_tot_value_range = F.elu(q_tot_value_range)

            x[:, :, k] = proj(q_tot_value_range, q_tot, self.v_min, self.v_max, self.n_atom)
        

        w_final = th.abs(self.hyper_w_final(states))         #[bs*episode_len, embed_dim]
        w_final = w_final.view(bs, -1, self.embed_dim, 1)    #[bs, episode_len, embed_dim, 1]
        v = self.V(states)                                   #[bs*episode_len, 1]
        v = v.view(bs, -1, 1)                                #[bs, episode_len, 1]

        value_range = self.value_range.unsqueeze(0).unsqueeze(0).expand(bs, episode_len, -1) #[bs, episode_len, n_atom]
        x_w = th.zeros_like(x, device=x.device)   #[bs, episode_len, embed_dim, n_atom]

        for i in range(self.embed_dim):
            next_v_range = value_range * w_final[:, :, i] #[bs, episode_len, n_atom]
            x_w[:, :, i] = proj(next_v_range, x[:, :, i], self.v_min, self.v_max, self.n_atom)

        q_tot = x_w[:, :, 0]
        for i in range(1, self.embed_dim):
            q_tot = bconv(q_tot, x_w[:, :, i])
        
        q_tot_value_range = self.q_tot_value_range2.unsqueeze(0).unsqueeze(0).expand(bs, episode_len, -1) #[bs, episode_len, N_ATOM]
        q_tot_value_range = q_tot_value_range + v

        q_tot = proj(q_tot_value_range, q_tot, self.v_min, self.v_max, self.n_atom)

        return q_tot  #[bs, episode_len, n_atom]

