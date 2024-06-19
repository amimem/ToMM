import torch
import random
import math
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, tanh
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
# from utils import MultiChannelNet

class MatchNet(nn.Module):
    def __init__(self,state_space_dim=16,hidden_size=128,num_ground_agents=10,num_abstract_agents=2,num_layers=2):
        super(MatchNet, self).__init__()
        self.state_space_dim = state_space_dim
        self.hidden_size = hidden_size
        self.num_ground_agents = num_ground_agents
        self.num_abstract_agents = num_abstract_agents
        self.num_layers = num_layers
        # self.encoder = {}
        self.encoder = []
        print(self.hidden_size)
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
            # self.encoder[abstract_agent] = nn.Sequential(mydict)
            self.encoder=nn.ModuleList(self.encoder)
        # self.encoder = MultiChannelNet(
        #         n_channels=self.num_abstract_agents,
        #         input_size=self.state_space_dim,
        #         hidden_layer_width=self.hidden_size,
        #         n_hidden_layers=self.num_layers,
        #         output_size=self.hidden_size,
        #     )

        self.assignment_matrix = nn.Parameter(torch.ones(self.num_ground_agents, self.num_abstract_agents)) # num_ground_agents x num_abstract_agents matrix of learnable paramaters w/ no iniitial bias
        
        # for model in self.encoder:
        #     for n, p in model.named_parameters():
        #         if "weight" in n:
        #             torch.nn.init.normal_(p, generator=rng)
        #         elif "bias" in n:
        #             torch.nn.init.normal_(p, generator=rng)


    def forward(self, state_batch, state_history_batch, joint_action_history_batch):
        # state_batch: batch_size x state_space_dim -> a batch of states for prediction of new joint actions
        # state_history_batch: batch_size x history_size x state_space_dim -> a batch of previous state histories associated with the time of each state in the batch
        # joint_action_history_batch: batch_size x history_size x num_ground_agents x action_dim -> a batch of previous joint action histories associated with the state histories
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
                # encoded_history = encoded_state_histories[abstract_agent][item] # history_size x hidden_size
                joint_action_history = joint_action_history_batch # history_size x num_ground_agents x action_dim
                # joint_action_history = joint_action_history_batch[item] # history_size x num_ground_agents x action_dim
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
        # print(predicted_joint_action_batch.shape)
        return predicted_joint_action_batch

# model = NewStompNet(state_space_dim=4,hidden_size=128,num_ground_agents=3,num_abstract_agents=2)
# loss_fn = nn.CrossEntropyLoss()
# opt = torch.optim.Adam(model.parameters(), lr=0.1)

# state0 = torch.tensor([0.0,0.0,0.0,0.0])
# state1 = torch.tensor([0.1,0.1,0.1,0.1])
# state2 = torch.tensor([0.2,0.2,0.2,0.2])
# state3 = torch.tensor([0.3,0.3,0.3,0.3])
# actions0 = torch.tensor([[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0]])
# actions1 = torch.tensor([[0.0,1.0,0.0,0.0],[0.0,1.0,0.0,0.0], [0.0,1.0,0.0,0.0]])
# actions2 = torch.tensor([[0.0,0.0,1.0,0.0],[0.0,0.0,1.0,0.0], [0.0,0.0,1.0,0.0]])
# actions3 = torch.tensor([[0.0,0.0,0.0,1.0],[0.0,0.0,0.0,1.0], [0.0,0.0,0.0,1.0]])

# state_batch = torch.cat([state0.unsqueeze(0),state3.unsqueeze(0)],axis=0)
# action_labels = torch.tensor([[0,0,0],[3,3,3]]).long()

# state_history = torch.cat([state1.unsqueeze(0),state2.unsqueeze(0)],axis=0)
# action_history_onehots = torch.cat([actions1.unsqueeze(0),actions2.unsqueeze(0)],axis=0)
# state_history_batch = torch.cat([state_history.unsqueeze(0), state_history.unsqueeze(0)],axis=0)
# action_history_onehots_batch = torch.cat([action_history_onehots.unsqueeze(0), action_history_onehots.unsqueeze(0)],axis=0)

# prediction = model.forward(state_batch,state_history_batch,action_history_onehots_batch)
# print("prediction:",prediction)

# loss = loss_fn(prediction.view(-1,4),action_labels.view(-1))
# print("loss:",loss)

# loss.backward()
# opt.step()

# prediction = model.forward(state_batch,state_history_batch,action_history_onehots_batch)
# print("prediction:",prediction)

# loss = loss_fn(prediction.view(-1,4),action_labels.view(-1))
# print("loss:",loss)