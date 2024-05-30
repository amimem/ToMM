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
        encoded_contexts = self.seq_enc(state_seq, actions_seq)# (bsz, seq_len, num_agents, enc_dim)
        action_logit_vectors = self.decoder.forward(
            torch.mean(encoded_contexts,dim=1) if self.decoder_type == 'BuffAtt' # (bsz, enc_dim)
            else encoded_contexts  # (bsz, num_agents, enc_dim)
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
        self.seq_model_step = nn.LSTMCell(
            self.enc_hidden_dim, self.enc_hidden_dim)
        if self.cross_talk:
            self.crosstalk_key = nn.Linear(self.enc_hidden_dim, self.enc_hidden_dim)
            self.crosstalk_query = nn.Linear(self.enc_hidden_dim, self.enc_hidden_dim)
            self.crosstalk_value = nn.Linear(self.enc_hidden_dim, self.enc_hidden_dim)
        self.fc_in = MLP(config.state_dim+config.num_actions,config.enc_MLPhidden_dim,self.enc_hidden_dim)
        self.fc_out = nn.Linear(self.enc_hidden_dim,config.enc_out_dim)

    def forward(self, state_seq, actions_seq):

        batch_size, seq_len, num_agents, num_actions = actions_seq.shape
        batch_size, seq_len, state_dim = state_seq.shape

        state_seq = torch.unsqueeze(state_seq, 2).repeat((1, 1, num_agents, 1))

        # seq_len, batch_size, num_agents,state_dim+num_actions
        input_seq = torch.transpose(
            torch.cat([state_seq, actions_seq], dim=-1), 0, 1).flatten(1, 2)
        input_seq = self.fc_in(input_seq)
        hx = torch.randn(batch_size*num_agents, self.enc_hidden_dim)
        cx = torch.randn(batch_size*num_agents, self.enc_hidden_dim)
        output_seq = []
        for step in range(seq_len):
            hx, cx = self.seq_model_step(input_seq[step], (hx, cx))
            if self.cross_talk:
                hx = hx + self.attn(hx)
            output_seq.append(hx.view((batch_size, num_agents, self.enc_hidden_dim)))
        output_seq = torch.transpose(torch.stack(output_seq, dim=0), 0, 1) # batch_size, seq_len, num_agents, enc_hidden_dim
        output_seq = self.fc_out(output_seq) # batch_size, seq_len, num_agents, enc_out_dim
        encoded_contexts = torch.mean(output_seq,dim=1)  # (bsz, num_agents, enc_out_dim)
        return encoded_contexts 

    def attn(self, h):
        Q = self.crosstalk_query(h)
        K = self.crosstalk_key(h)
        V = self.crosstalk_value(h)
        return F.softmax(Q @ K.T/np.sqrt(self.enc_hidden_dim),dim=-1) @ V


class BufferAttentionDecoder():
    def __init__(self, config):
        super().__init__()
        self.bufferoutput_dim = (config.num_agents, config.num_actions)
        self.context_history = None
        self.joint_action_history = None
        self.current_context_batch = 0

    def append_to_buffer(self, target_actions):
        self.context_history = self.current_context_batch.detach().clone() if self.context_history is None else torch.cat(
            (self.context_history, self.current_context_batch.detach().clone()), dim=0)  # history_size x hidden_size
        action_onehots_batch = torch.nn.functional.one_hot(target_actions.detach().clone(),
                                                           num_classes=self.bufferoutput_dim[1]).float()
        self.joint_action_history = action_onehots_batch if self.joint_action_history is None else torch.cat([
            self.joint_action_history, action_onehots_batch], dim=0)  # history_size x num_ground_agents x action_dim

    def forward(self, context_samples):
        batch_size, enc_dim = context_samples.shape
        self.current_context_batch = context_samples
        if self.context_history == None:
            predicted_joint_actions = F.softmax(torch.ones(
                (batch_size, self.bufferoutput_dim[0], self.bufferoutput_dim[1])),dim=-1)
        else:
            predicted_joint_actions = []
            for i in range(batch_size):
                attention_weights = F.softmax(torch.matmul(
                    self.context_history, self.current_context_batch[i]), dim=-1)  # history_size
                abs_predicted_joint_action = torch.matmul(self.joint_action_history.permute(
                    -2, -1, 0), attention_weights)  # num_ground_agents x action_dim
                predicted_joint_actions.append(
                    abs_predicted_joint_action.unsqueeze(0))
            # batch_size x num_ground_agents x action_dim
            predicted_joint_actions = torch.cat(
                predicted_joint_actions, axis=0)
        # print(predicted_joint_actions.shape)
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
    solve_quadratic = lambda a, b, c: -b+np.sqrt(b**2-4*a*c)/(2*a)
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
    elif v.model_name == 'MLP':
        a = n_layers
        b = v.state_dim+v.num_actions
        c = -v.P/v.num_agents
        W = solve_quadratic(a, b, c)
    elif v.model_name == 'sharedMLP':
        a = n_layers
        b = (v.seq_len+1)*v.num_actions +v.seq_len*v.state_dim
        c = -v.P
        W = solve_quadratic(a, b, c)
    else:
        print('choose valid model name')
    if W<2:
        W=2
    return W


#------------------illustrative MLP baselines--------------------------

class MLPperagent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W = int(get_width(config))
        config.enc_out_dim = self.W
        self.model = nn.ModuleList([MLP(config.state_dim, config.enc_out_dim, config.num_actions) 
            for agent in range(config.num_agents)])

    def forward(self, state_seq, actions_seq):
        # state_seq: (bsz, seq_len, state_dim)
        # actions_seq: (bsz, seq_len, num_agents, num_actions)
        current_state = torch.squeeze(state_seq[:,-1,:])
        batch_size, seq_len, num_agents, num_actions = actions_seq.shape
        action_logits = []
        for agent in range(num_agents):
            action_logits.append(self.model[agent](current_state).unsqueeze(dim=1)) # (batch_size,1,num_actions)
        return torch.cat(action_logits,dim=1) # (batch_size,num_agents,num_actions)

    def count_parameters(self,):
        assert isinstance(self, nn.Module), "model must be a torch.nn.Module"
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPallagents(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W = int(get_width(config))
        config.enc_out_dim = self.W
        self.model = MLP(config.state_dim, config.enc_out_dim, config.num_actions*config.num_agents) 

    def forward(self, state_seq, actions_seq):
        # state_seq: (bsz, seq_len, state_dim)
        # actions_seq: (bsz, seq_len, num_agents, num_actions)
        current_state = torch.squeeze(state_seq[:,-1,:])
        
        output=self.model(current_state).view((batch_size,num_agents, num_actions))
        return output

    def count_parameters(self,):
        assert isinstance(self, nn.Module), "model must be a torch.nn.Module"
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class sharedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W = int(get_width(config))
        config.enc_out_dim=self.W
        singleagent_context_size = config.seq_len*(config.state_dim + config.num_actions)
        self.model = MLP(singleagent_context_size, config.enc_out_dim, config.num_actions)

    def forward(self,state_seq, actions_seq):
        # state_seq: (bsz, seq_len, state_dim)
        # actions_seq: (bsz, seq_len, num_agents, num_actions)
        batch_size, seq_len, num_agents, num_actions = actions_seq.shape
        action_logits = []
        flattened_context = torch.cat((
            torch.unsqueeze(torch.flatten(state_seq,start_dim=1),1).repeat((1,num_agents,1)), #(bsz, num_agents, seq_len*state_dim)
            torch.flatten(torch.transpose(actions_seq,1,2),start_dim=2) #(bsz, num_agents, seq_len*num_actions)
            ),dim=2).view(batch_size*num_agents,-1) #(bsz*num_agents, seq_len*(state_dim +num_actions))
        output=self.model(flattened_context).view(*(batch_size, num_agents, num_actions))
        return output

    def count_parameters(self,):
        assert isinstance(self, nn.Module), "model must be a torch.nn.Module"
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# class MatchNet(nn.Module):
#     def __init__(self,state_dim=16,hidden_size=128,num_ground_agents=10,num_abstract_agents=2,num_layers=2):
#         super(MatchNet, self).__init__()
#         self.name='match'
#         self.state_dim = state_dim
#         self.hidden_size = hidden_size
#         self.num_ground_agents = num_ground_agents
#         self.num_abstract_agents = num_abstract_agents
#         self.num_layers = num_layers
#         self.encoder = []
#         for abstract_agent in range(self.num_abstract_agents):
#             layers = [("linear0",nn.Linear(self.state_dim,self.hidden_size)), ("relu0",nn.ReLU())]
#             if self.num_layers > 1:
#                 for i in range(self.num_layers-1):
#                     layers.append(("linear"+str(i+1), nn.Linear(self.hidden_size,self.hidden_size)))
#                     layers.append(("relu"+str(i+1),nn.ReLU()))
#                 layers.append(("linear"+str(self.num_layers+1), nn.Linear(self.hidden_size,self.hidden_size)))
#                 layers.append(("relu"+str(self.num_layers+1),nn.ReLU()))
#             mydict = OrderedDict(layers)
#             self.encoder.append(nn.Sequential(mydict))
#             self.encoder=nn.ModuleList(self.encoder)
#         self.assignment_matrix = nn.Parameter(torch.ones(self.num_ground_agents, self.num_abstract_agents)) # num_ground_agents x num_abstract_agents matrix of learnable paramaters w/ no iniitial bias

#     def forward(self, state_batch, state_history_batch, joint_action_history_batch):
#         encoded_states = {}
#         encoded_state_histories = {}
#         for abstract_agent in range(self.num_abstract_agents):
#             encoded_states[abstract_agent] = self.encoder[abstract_agent](state_batch) # batch_size x hidden_size
#             encoded_state_histories[abstract_agent] = self.encoder[abstract_agent](state_history_batch) # batch_size x history_size x hidden_size

#         batch_size = state_batch.size()[0]
#         predicted_joint_actions = []
#         for item in range(batch_size):
#             abs_predicted_joint_actions = []
#             for abstract_agent in range(self.num_abstract_agents):
#                 encoded_state = encoded_states[abstract_agent][item] # hidden_size
#                 encoded_history = encoded_state_histories[abstract_agent] # history_size x hidden_size
#                 joint_action_history = joint_action_history_batch # history_size x num_ground_agents x action_dim
#                 attention_weights = F.softmax(torch.matmul(encoded_history,encoded_state),dim=-1) # history_size
#                 abs_predicted_joint_action = torch.matmul(joint_action_history.permute(-2,-1,0), attention_weights)#.T # num_ground_agents x action_dim
#                 abs_predicted_joint_actions.append(abs_predicted_joint_action.unsqueeze(0))

#             if self.num_abstract_agents>1:
#                 abs_predicted_joint_actions = torch.cat(abs_predicted_joint_actions, axis=0).transpose(1,0) # num_ground_agents x num_abstract_agents x action_dim
#                 soft_assignments = F.softmax(self.assignment_matrix, dim = -1) # num_ground_agents x num_abstract_agents
#                 predicted_joint_action = torch.bmm(soft_assignments.unsqueeze(1), abs_predicted_joint_actions).squeeze(1) # num_ground_agents x action_dim
#                 predicted_joint_actions.append(predicted_joint_action.unsqueeze(0))
#             else:
#                 predicted_joint_actions.append(abs_predicted_joint_action.unsqueeze(0))

#         predicted_joint_action_batch = torch.cat(predicted_joint_actions, axis=0) # batch_size x num_ground_agents x action_dim
#         return predicted_joint_action_batch


# def init_model(model_config):

#     context_size = model_config['context_size']
#     state_dim = model_config['state_dim']
#     num_actions = model_config['num_actions']
#     num_agents = model_config['num_agents']
#     hidden_size = model_config['hidden_size']
#     num_hidden_layers = model_config['num_hidden_layers']

#     if model_config['model_name']=='mlp':
#         input_size=context_size+state_dim
#         model = MLP(input_size, hidden_size, num_actions, num_hidden_layers)
#     elif model_config['model_name']=='match':
#         num_groups = model_config['M']
#         model = MatchNet(
#             state_dim=state_dim,
#             hidden_size=hidden_size,
#             num_ground_agents=num_agents,
#             num_abstract_agents=num_groups,
#             num_layers=num_hidden_layers,
#         )
#     else:
#         abort("Choose valid model name")

#     return model

       # if i==0: #waste first batch to initialize training memory.
       #     self.context_history=context_sample.detach().clone()
       #     joint_action_history=torch.nn.functional.one_hot(
       #         target_actions, num_classes=dataloader.dataset.num_actions).float()
       #     continue
       # else:
       #     action_logit_vectors = model.matching(self.context_history,self.joint_action_history)

       # state_dim=16,hidden_size=128,num_ground_agents=10,num_abstract_agents=2,num_layers=2):
