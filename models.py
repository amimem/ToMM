import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn.functional as F

def init_model(model_config):

    context_size = model_config['context_size']
    state_space_dim = model_config['state_space_dim']
    num_actions = model_config['num_actions']
    num_agents = model_config['num_agents']
    hidden_size = model_config['hidden_size']
    num_hidden_layers = model_config['num_hidden_layers']

    if model_config['model_name']=='mlp':
        input_size=context_size+state_space_dim
        model = MLP(input_size, hidden_size, num_actions, num_hidden_layers)
    elif model_config['model_name']=='match':
        num_groups = model_config['M']
        model = MatchNet(
            state_space_dim=state_space_dim,
            hidden_size=hidden_size,
            num_ground_agents=num_agents,
            num_abstract_agents=num_groups,
            num_layers=num_hidden_layers,
        )     
    else:
        abort("Choose valid model name")

    return model


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_hidden_layers)],
            nn.Linear(hidden_size, output_size)
        )
        self.name='mlp'

    def forward(self, x):
        return self.model(x)


class MatchNet(nn.Module):
    def __init__(self,state_space_dim=16,hidden_size=128,num_ground_agents=10,num_abstract_agents=2,num_layers=2):
        super(MatchNet, self).__init__()
        self.name='match'
        self.state_space_dim = state_space_dim
        self.hidden_size = hidden_size
        self.num_ground_agents = num_ground_agents
        self.num_abstract_agents = num_abstract_agents
        self.num_layers = num_layers
        self.encoder = []
        for abstract_agent in range(self.num_abstract_agents):
            layers = [("linear0",nn.Linear(self.state_space_dim,self.hidden_size)), ("relu0",nn.ReLU())]
            if self.num_layers > 1:
                for i in range(self.num_layers-1):
                    layers.append(("linear"+str(i+1), nn.Linear(self.hidden_size,self.hidden_size)))
                    layers.append(("relu"+str(i+1),nn.ReLU()))
                layers.append(("linear"+str(self.num_layers+1), nn.Linear(self.hidden_size,self.hidden_size)))
                layers.append(("relu"+str(self.num_layers+1),nn.ReLU()))
            mydict = OrderedDict(layers)
            self.encoder.append(nn.Sequential(mydict))
            self.encoder=nn.ModuleList(self.encoder)
        self.assignment_matrix = nn.Parameter(torch.ones(self.num_ground_agents, self.num_abstract_agents)) # num_ground_agents x num_abstract_agents matrix of learnable paramaters w/ no iniitial bias
        
    def forward(self, state_batch, state_history_batch, joint_action_history_batch):
        encoded_states = {}
        encoded_state_histories = {}
        for abstract_agent in range(self.num_abstract_agents):
            encoded_states[abstract_agent] = self.encoder[abstract_agent](state_batch) # batch_size x hidden_size
            encoded_state_histories[abstract_agent] = self.encoder[abstract_agent](state_history_batch) # batch_size x history_size x hidden_size

        batch_size = state_batch.size()[0]
        predicted_joint_actions = []
        for item in range(batch_size):
            abs_predicted_joint_actions = []
            for abstract_agent in range(self.num_abstract_agents):
                encoded_state = encoded_states[abstract_agent][item] # hidden_size
                encoded_history = encoded_state_histories[abstract_agent] # history_size x hidden_size
                joint_action_history = joint_action_history_batch # history_size x num_ground_agents x action_dim
                attention_weights = F.softmax(torch.matmul(encoded_history,encoded_state),dim=-1) # history_size
                abs_predicted_joint_action = torch.matmul(joint_action_history.permute(-2,-1,0), attention_weights)#.T # num_ground_agents x action_dim
                abs_predicted_joint_actions.append(abs_predicted_joint_action.unsqueeze(0))
            
            if self.num_abstract_agents>1:
                abs_predicted_joint_actions = torch.cat(abs_predicted_joint_actions, axis=0).transpose(1,0) # num_ground_agents x num_abstract_agents x action_dim
                soft_assignments = F.softmax(self.assignment_matrix, dim = -1) # num_ground_agents x num_abstract_agents
                predicted_joint_action = torch.bmm(soft_assignments.unsqueeze(1), abs_predicted_joint_actions).squeeze(1) # num_ground_agents x action_dim
                predicted_joint_actions.append(predicted_joint_action.unsqueeze(0))
            else:
                predicted_joint_actions.append(abs_predicted_joint_action.unsqueeze(0))

        predicted_joint_action_batch = torch.cat(predicted_joint_actions, axis=0) # batch_size x num_ground_agents x action_dim
        return predicted_joint_action_batch