import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

class STOMP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder_type = config.decoder_type
        # self.state_dim = config.state_dim
        # self.num_actions = config.num_actions
        # self.num_agents = config.num_agents

        self.W = int(get_width(config))
        config.enc_MLPhidden_dim = self.W
        config.enc_hidden_dim = self.W
        config.enc_out_dim = self.W
        if config.decoder_type=='MLP':
            config.dec_hidden_dim = self.W

        print(f"enc_hidden/enc_out dims: {config.enc_hidden_dim}/{config.enc_out_dim}")

        # modules
        self.seq_enc = SeqEnc(config, inter_model_type=config.inter_model_type) # (bsz,num_agents,seq_len) to (bsz,num_agents,seq_len, enc_dim)
        self.decoder = BufferAttentionDecoder(config) if self.decoder_type == 'BuffAtt' else MLP(
            config.enc_out_dim, config.dec_hidden_dim, config.num_actions)

    def forward(self, state_seq, actions_seq):
        # state_seq: (bsz, seq_len, state_dim)
        # actions_seq: (bsz, seq_len, num_agents, num_actions)
        encoded_contexts = self.seq_enc(state_seq, actions_seq) # (bsz, num_agents, enc_dim)
        # print(encoded_contexts.shape)
        action_logit_vectors = self.decoder.forward(
            torch.mean(encoded_contexts,dim=1) if self.decoder_type == 'BuffAtt' # (bsz, enc_dim)
            else encoded_contexts  
        )
        return action_logit_vectors

    def count_parameters(self,):
        assert isinstance(self, nn.Module), "model must be a torch.nn.Module"
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SeqEnc(nn.Module):
    # MLP-map the states&actions, sequence process, average over steps, and finally MLP-map to latent space
    def __init__(self, config,inter_model_type=None):
        super().__init__()
        self.enc_hidden_dim = config.enc_hidden_dim
        self.fc_in = MLP(config.state_dim+config.num_actions,config.enc_MLPhidden_dim,self.enc_hidden_dim)

        #modules for sequence model
        self.LSTM = nn.LSTM(self.enc_hidden_dim, self.enc_hidden_dim)

        self.pe = self.positionalencoding1d(self.enc_hidden_dim,config.num_agents)

        #modules for interaction model
        self.inter_model_type = inter_model_type
        if inter_model_type is not None:
            self.d = self.enc_hidden_dim
            self.d_I = self.enc_hidden_dim
            self.num_inducing_points = 10

            if self.inter_model_type == 'attn':
                self.attn_fcs1 = nn.ModuleList([
                    nn.Linear(self.d_I, self.d),
                    nn.Linear(self.enc_hidden_dim, self.d),
                    nn.Linear(self.enc_hidden_dim, self.d)
                ])
            elif self.inter_model_type == 'ipattn':
                self.attn_fcs1 = nn.ModuleList([
                    nn.Linear(self.d_I, self.d),
                    nn.Linear(self.enc_hidden_dim, self.d),
                    nn.Linear(self.enc_hidden_dim, self.d)
                ])
                self.attn_fcs2 = nn.ModuleList([
                    nn.Linear(self.d_I, self.d),
                    nn.Linear(self.enc_hidden_dim, self.d),
                    nn.Linear(self.enc_hidden_dim, self.d)
                ])
                self.ips = nn.Embedding(num_embeddings=self.num_inducing_points, embedding_dim=self.d_I)
            elif self.inter_model_type == 'SAB':
                self.SAB = nn.Sequential(
                    SAB(self.d, self.d, 1, ln=False),
                    SAB(self.d, self.d, 1, ln=False)
                )            
            elif self.inter_model_type == 'ISAB':
                self.ISAB = ISAB(
                    self.d, self.d, 1, self.num_inducing_points, ln=False
                )
            elif self.inter_model_type == 'rnn': #biLSTM':
                self.seq_model_agent = nn.RNN(
                    self.enc_hidden_dim, int(self.enc_hidden_dim), batch_first=True
                )
                    # self.enc_hidden_dim, int(self.enc_hidden_dim/2), batch_first=True,bidirectional=True)
            elif self.inter_model_type is None:
                pass
            else:
                print("invalid interaction model type")

        # self.fc_out = nn.Linear(self.enc_hidden_dim,config.enc_out_dim)

    def sequence_model(self, x):
        _, (h_final, _) = self.LSTM(x)
        return h_final[-1]

    def agent_interaction_model(self, x):
        batch_size = x.shape[0] #batch_size, num_agents, enc_hidden_dim

        if self.inter_model_type == 'attn':
            # attention (compute is quadratic in num agents)
            return x+ self.attn(x,x)
        elif self.inter_model_type == 'ipattn':
            # inducing point attention (compute is linear in num agents)
            return x+self.attn(x,self.attn(self.ips.weight.repeat((batch_size,1,1)),x,self.attn_fcs1),self.attn_fcs2) # TODO: replace repeat with more efficient broadcasting
        elif self.inter_model_type == 'SAB':
            return self.SAB(x)
        elif self.inter_model_type == 'ISAB':
            return self.ISAB(x)
        elif self.inter_model_type == 'rnn':
            hx = torch.randn(1, batch_size, int(self.enc_hidden_dim))
            cx = torch.randn(1, batch_size, int(self.enc_hidden_dim)) 
            # hx = torch.randn(2, batch_size, int(self.enc_hidden_dim/2))
            # cx = torch.randn(2, batch_size, int(self.enc_hidden_dim/2))      
            # x, _ = self.seq_model_agent(x, (hx, cx))
            x, _ = self.seq_model_agent(x, hx)
            return x
        else:
            print("invalid interaction model type")

    def attn(self, x, y, fcs):
        Q = fcs[0](x)
        K = fcs[1](y)
        V = fcs[2](y)
        return F.softmax(Q @ K.transpose(-1,-2)/np.sqrt(self.d),dim=-1) @ V

    def forward(self, state_seq, actions_seq):

        #reshape and concatentate states and actions
        batch_size, seq_len, num_agents, num_actions = actions_seq.shape
        # batch_size, seq_len, state_dim = state_seq.shape
        state_seq = torch.unsqueeze(state_seq, 2).repeat((1, 1, num_agents, 1))
        # batch_size, seq_len, num_agents, state_dim = state_seq.shape
        x = torch.transpose(
            torch.cat([state_seq, actions_seq], dim=-1), 0, 1)
        # seq_len, batch_size, num_agents, state_dim + num_actions
        
        #process
        x = self.fc_in(x.flatten(1, 2)) # seq_len, batch_size*num_agents, enc_hidden_dim
        x = self.sequence_model(x).view((batch_size, num_agents, self.enc_hidden_dim)) # batch_size,num_agents, enc_hidden_dim
        # x = x + self.pe.unsqueeze(dim=0).repeat((batch_size,1,1))
        if self.inter_model_type is not None:
            x = self.agent_interaction_model(x)
        # x = self.fc_out(x) # batch_size, num_agents, enc_out_dim

        return x


    def positionalencoding1d(self,d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(torch.log(torch.tensor(10000.0)) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class BufferAttentionDecoder():
    def __init__(self, config):
        super().__init__()
        self.bufferoutput_dim = (config.num_agents, config.num_actions)
        self.latent_history = None
        self.joint_action_history = None
        self.current_latent_batch = 0

    def append_to_buffer(self, target_actions):
        self.latent_history = self.current_latent_batch.detach().clone() if self.latent_history is None else torch.cat(
            (self.latent_history, self.current_latent_batch.detach().clone()), dim=0)  # history_size x hidden_size
        action_onehots_batch = torch.nn.functional.one_hot(target_actions.detach().clone(),
                                                           num_classes=self.bufferoutput_dim[1]).float()
        self.joint_action_history = action_onehots_batch if self.joint_action_history is None else torch.cat([
            self.joint_action_history, action_onehots_batch], dim=0)  # history_size x num_ground_agents x action_dim

    def forward(self, latent_avg_samples):
        batch_size, enc_dim = latent_avg_samples.shape
        self.current_latent_batch = latent_avg_samples # batch_size x enc_dim
        if self.latent_history == None:
            predicted_joint_actions = F.softmax(torch.ones(
                (batch_size, self.bufferoutput_dim[0], self.bufferoutput_dim[1])),dim=-1)
        else:
            predicted_joint_actions = []
            for i in range(batch_size):
                attention_weights = F.softmax(torch.matmul(
                    self.latent_history, self.current_latent_batch[i]), dim=-1)  # history_size
                abs_predicted_joint_action = torch.matmul(self.joint_action_history.permute(
                    -2, -1, 0), attention_weights)  # num_ground_agents x action_dim
                predicted_joint_actions.append(
                    abs_predicted_joint_action.unsqueeze(0))
            
            predicted_joint_actions = torch.cat(
                predicted_joint_actions, axis=0) # batch_size x num_ground_agents x action_dim
        return predicted_joint_actions


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_hidden_layers)],
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

def get_width(v):
    solve_quadratic = lambda a, b, c: (-b+np.sqrt(b**2-4*a*c))/(2*a)
    n_layers = 2
    if v.model_name == 'STOMP':
        # if v.decoder_type == 'MLP' and v.cross_talk:
        a = (2*n_layers+1+8+3+1) #17
        if v.inter_model_type==None:
            a = a - 3
        b = (v.num_actions+v.state_dim)+4+v.num_actions
        c = -v.P
        W = solve_quadratic(a, b, c)
    elif v.model_name == 'MLP_nosharing':
        a = n_layers
        b = v.state_dim+v.num_actions
        c = -v.P/v.num_agents
        W = solve_quadratic(a, b, c)
    elif v.model_name == 'MLP_encodersharingonly':
        a = n_layers
        b = v.state_dim+v.num_actions*v.num_agents
        c = -v.P
        W = solve_quadratic(a, b, c)
    elif v.model_name == 'MLP_fullsharing':
        print(v.model_name)
        a = n_layers
        b = v.state_dim +v.seq_len*(v.num_actions +v.state_dim)
        c = -v.P
        W = solve_quadratic(a, b, c)
    else:
        print('choose valid model name')
    minimum_capacity = 2
    if W<minimum_capacity:
        W=minimum_capacity
    W=int(W)
    if (W%2)!=0:
        W = W+1
    return W
#------------------data generation models

class logit(nn.Module):
    def __init__(self, config,rng):
        super().__init__()

        assert config.num_actions ==2, "implemented for num_action = 2"
        mean = np.zeros(config.num_agents)
        cov = config.corr*np.ones((config.num_agents,config.num_agents))+\
                (1-config.corr)*np.eye(config.num_agents)
        num_obs = 2**config.state_dim
        action_at_corr1_logits = rng.multivariate_normal(mean,cov,size=num_obs) # num_samples,num_agents
        # introduce label disorder for this action
        self.action_at_corr1 = rng.integers(0,high=config.num_actions,size=config.num_agents)
        # (Assuming action selection based on sign of logit)
        # action 0 logit is same as action at corr=1 logit if action at corr=1 is 0, else it is the negative of action logit at corr=1
        self.action_0_logits = action_at_corr1_logits * np.power(-1,self.action_at_corr1)[np.newaxis,:]
        self.mask = config.num_actions**np.arange(config.state_dim)
        self.state_fn = rng.integers(0,high=config.state_dim,size=num_obs)

    def forward(self, state):
        # dim(state): batch_size, state_dim
        obs=(state>0) # batch_size, state_dim
        state_idx = self.obs2index(obs)
        agent_component = self.action_0_logits[state_idx]
        state_component = state[self.state_fn[state_idx]]
        agent_weight_factor = 1
        action_0_logits = (state_component + agent_weight_factor*agent_component)/np.sqrt(1+agent_weight_factor)
        return np.vstack([action_0_logits,-action_0_logits]).T

    def obs2index(self, obs):
        return np.sum(self.mask * obs[np.newaxis,:],-1)

#------------------illustrative MLP baselines--------------------------

class MLPbaselines(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W = int(get_width(config))
        print(f"hidden_dim: {self.W}")
        config.enc_out_dim = self.W

        self.model_name = config.model_name
        if self.model_name == 'MLP_nosharing':
            self.model = nn.ModuleList([MLP(config.state_dim, config.enc_out_dim, config.num_actions) 
                for agent in range(config.num_agents)])
        elif self.model_name == 'MLP_fullsharing':
            singleagent_context_size = config.seq_len*(config.state_dim + config.num_actions)
            self.model = MLP(singleagent_context_size, config.enc_out_dim, config.num_actions)
        elif self.model_name == 'MLP_encodersharingonly':
            self.model = MLP(config.state_dim, config.enc_out_dim, config.num_agents*config.num_actions) 
        else:
            print('select valid MLP baseline type')

    def forward(self, state_seq, actions_seq):
        # state_seq: (bsz, seq_len, state_dim)
        # actions_seq: (bsz, seq_len, num_agents, num_actions)

        current_state = torch.squeeze(state_seq[:,-1,:])
        batch_size, seq_len, num_agents, num_actions = actions_seq.shape

        if self.model_name=='MLP_nosharing':
            action_logits = []
            for agent in range(num_agents):
                action_logits.append(self.model[agent](current_state).unsqueeze(dim=1)) # (batch_size,1,num_actions)
            output=torch.cat(action_logits,dim=1) # (batch_size,num_agents,num_actions)
        elif self.model_name == 'MLP_fullsharing':
            flattened_context = torch.cat((
                torch.unsqueeze(torch.flatten(state_seq,start_dim=1),1).repeat((1,num_agents,1)), #(bsz, num_agents, seq_len*state_dim)
                torch.flatten(torch.transpose(actions_seq,1,2),start_dim=2) #(bsz, num_agents, seq_len*num_actions)
                ),dim=2).view(batch_size*num_agents,-1) #(bsz*num_agents, seq_len*(state_dim +num_actions))
            output=self.model(flattened_context).view(*(batch_size, num_agents, num_actions))
        elif self.model_name == 'MLP_encodersharingonly':
            output=self.model(current_state).view((batch_size, num_agents, num_actions))
        else:
            print('select valid MLP baseline type')

        return output

    def count_parameters(self,):
        assert isinstance(self, nn.Module), "model must be a torch.nn.Module"
        return sum(p.numel() for p in self.parameters() if p.requires_grad)