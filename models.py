import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

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
        self.seq_enc = SeqEnc(config, cross_talk=config.cross_talk) # (bsz,num_agents,seq_len) to (bsz,num_agents,seq_len, enc_dim)
        self.decoder = BufferAttentionDecoder(config) if self.decoder_type == 'BuffAtt' else MLP(
            config.enc_out_dim, config.dec_hidden_dim, config.num_actions)

    def forward(self, state_seq, actions_seq):
        # state_seq: (bsz, seq_len, state_dim)
        # actions_seq: (bsz, seq_len, num_agents, num_actions)
        encoded_contexts = self.seq_enc(state_seq, actions_seq) # (bsz, num_agents, enc_dim)
        
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
    def __init__(self, config,cross_talk=False):
        super().__init__()
        self.enc_hidden_dim = config.enc_hidden_dim
        self.cross_talk = cross_talk
        self.fc_in = MLP(config.state_dim+config.num_actions,config.enc_MLPhidden_dim,self.enc_hidden_dim)

        #modules for sequence model
        self.LSTM = nn.LSTM(self.enc_hidden_dim, self.enc_hidden_dim)
        #modules for interaction model
        self.d = self.enc_hidden_dim
        self.d_I = self.enc_hidden_dim
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
        self.num_inducing_points = 10
        self.ips = nn.Embedding(num_embeddings=self.num_inducing_points, embedding_dim=self.d)

        # self.fc_out = nn.Linear(self.enc_hidden_dim,config.enc_out_dim)

    def sequence_model(self, x):
        _, (h_final, _) = self.LSTM(x)
        return h_final[0]


    def agent_interaction_model(self, x):
        batch_size = x.shape[0]
        # # attention (quadratic compute in num agents)
        # return self.attn(x,x)
        # inducing point attention (linear compute in num agents)
        return self.attn(x,self.attn(self.ips.weight.repeat((batch_size,1,1)),x,self.attn_fcs1),self.attn_fcs2) # TODO: replace repeat with more efficient broadcasting

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
            torch.cat([state_seq, actions_seq], dim=-1), 0, 1).flatten(1, 2)
        # seq_len, batch_size*num_agents, state_dim + num_actions
        
        #process
        x = self.fc_in(x) # seq_len, batch_size*num_agents, enc_hidden_dim
        x = self.sequence_model(x) # batch_size*num_agents, enc_hidden_dim
        if self.cross_talk:
            x = self.agent_interaction_model(x.view((batch_size, num_agents, self.enc_hidden_dim)))
        # x = self.fc_out(x) # batch_size, num_agents, enc_out_dim

        return x


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
        a = 17
        b = v.seq_len*(v.num_actions+v.state_dim)+4*v.num_actions
        c = -v.P
        W = solve_quadratic(a, b, c)
        # elif v.decoder_type != 'BuffAtt' and v.cross_talk:
        #     a = (num_hidden_layers+1)*v.M_model
        #     b = v.M_model*(v.state_dim)
        #     c = v.num_agents*v.M_model-v.P
        #     W = solve_quadratic(a, b, c)
        # elif ecoder_type == 'BuffAtt' and ~v.cross_talk:
        #     a = (num_hidden_layers+1)*v.M_model
        #     b = v.M_model*(v.state_dim)
        #     c = v.num_agents*v.M_model-v.P
        #     W = solve_quadratic(a, b, c)
        # elif v.decoder_type != 'BuffAtt' and ~v.cross_talk:
        #             a = (num_hidden_layers+1)*v.M_model
        #     b = v.M_model*(v.state_dim)
        #     c = v.num_agents*v.M_model-v.P
        #     W = solve_quadratic(a, b, c)
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

    def forward(self, state):
        obs=(state>0) # batch_size, state_dim
        action_0_logits = self.action_0_logits[self.obs2index(obs)]
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