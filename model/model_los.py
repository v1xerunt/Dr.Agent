import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

class Agent(nn.Module):
    def __init__(self, cell='gru', use_baseline=True, n_actions=10, n_units=64, fusion_dim=128, n_input=76, n_hidden=128, demo_dim=17, n_output=1, dropout=0.0, lamda=0.5, device='cpu'):
        super(Agent, self).__init__()

        self.cell = cell
        self.use_baseline = use_baseline
        self.n_actions = n_actions
        self.n_units = n_units
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dropout = dropout
        self.lamda = lamda
        self.fusion_dim = fusion_dim
        self.demo_dim = demo_dim
        self.device = device

        self.agent1_action = []
        self.agent1_prob = []
        self.agent1_entropy = []
        self.agent1_baseline = []
        self.agent2_action = []
        self.agent2_prob = []
        self.agent2_entropy = []
        self.agent2_baseline = []

        self.agent1_fc1 = nn.Linear(self.n_hidden + self.demo_dim, self.n_units)
        self.agent2_fc1 = nn.Linear(self.n_input + self.demo_dim, self.n_units)
        self.agent1_fc2 = nn.Linear(self.n_units, self.n_actions)
        self.agent2_fc2 = nn.Linear(self.n_units, self.n_actions)
        if use_baseline == True:
            self.agent1_value = nn.Linear(self.n_units, 1)
            self.agent2_value = nn.Linear(self.n_units, 1)
        
        if self.cell == 'lstm':
            self.rnn = nn.LSTMCell(self.n_input, self.n_hidden)
        else:
            self.rnn = nn.GRUCell(self.n_input, self.n_hidden)
            
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        
        if dropout > 0.0:
            self.nn_dropout = nn.Dropout(p=dropout)
        self.init_h = nn.Linear(self.demo_dim, self.n_hidden)
        self.init_c = nn.Linear(self.demo_dim, self.n_hidden)
        self.fusion = nn.Linear(self.n_hidden+self.demo_dim, self.fusion_dim)
        self.output = nn.Linear(self.fusion_dim, self.n_output)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def choose_action(self, observation, agent=1):
        observation = observation.detach()
        
        if agent == 1:
            result_fc1 = self.agent1_fc1(observation)
            result_fc1 = self.tanh(result_fc1)
            result_fc2 = self.agent1_fc2(result_fc1)
            if self.use_baseline == True:
                result_value = self.agent1_value(result_fc1)
                self.agent1_baseline.append(result_value)
        else:
            result_fc1 = self.agent2_fc1(observation)
            result_fc1 = self.tanh(result_fc1)
            result_fc2 = self.agent2_fc2(result_fc1)
            if self.use_baseline == True:
                result_value = self.agent2_value(result_fc1)
                self.agent2_baseline.append(result_value)
            
        probs = self.softmax(result_fc2)
        m = torch.distributions.Categorical(probs)
        actions = m.sample()
        
        if agent == 1:
            self.agent1_entropy.append(m.entropy())
            self.agent1_action.append(actions.unsqueeze(-1))
            self.agent1_prob.append(m.log_prob(actions))
        else:
            self.agent2_entropy.append(m.entropy())
            self.agent2_action.append(actions.unsqueeze(-1))
            self.agent2_prob.append(m.log_prob(actions))
            
        return actions.unsqueeze(-1)

    def forward(self, input, demo):
        # input shape [batch_size, timestep, feature_dim]
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)

        # Initialization
        self.agent1_action = []
        self.agent1_prob = []
        self.agent1_entropy = []
        self.agent1_baseline = []
        self.agent2_action = []
        self.agent2_prob = []
        self.agent2_entropy = []
        self.agent2_baseline = []
        
        cur_h = self.init_h(demo)
        if self.cell == 'lstm':
            cur_c = self.init_c(demo)
            
        h = []

        for cur_time in range(time_step):
            cur_input = input[:, cur_time, :]
            
            if cur_time == 0:
                obs_1 = cur_h
                obs_2 = cur_input
                obs_1 = torch.cat((obs_1, demo), dim=1)
                obs_2 = torch.cat((obs_2, demo), dim=1)
                self.choose_action(obs_1, 1).long()
                self.choose_action(obs_2, 2).long()

                observed_h = torch.zeros_like(cur_h, dtype=torch.float32).view(-1).repeat(
                    self.n_actions).view(self.n_actions, batch_size, self.n_hidden)
                action_h = cur_h
                if self.cell == 'lstm':
                    observed_c = torch.zeros_like(cur_c, dtype=torch.float32).view(-1).repeat(self.n_actions).view(self.n_actions, batch_size, self.n_hidden)
                    action_c = cur_c
                
            else:
                observed_h = torch.cat((observed_h[1:], cur_h.unsqueeze(0)), 0)
                obs_1 = cur_h
                obs_2 = cur_input
                obs_1 = torch.cat((obs_1, demo), dim=1)
                obs_2 = torch.cat((obs_2, demo), dim=1)
                act_idx1 = self.choose_action(obs_1, 1).long()
                act_idx2 = self.choose_action(obs_2, 2).long()
                batch_idx = torch.arange(batch_size, dtype=torch.long).unsqueeze(-1).to(self.device)
                action_h1 = observed_h[act_idx1, batch_idx, :].squeeze(1)
                action_h2 = observed_h[act_idx2, batch_idx, :].squeeze(1)
                action_h = (action_h1 + action_h2) / 2
                if self.cell == 'lstm':
                    observed_c = torch.cat((observed_c[1:], cur_c.unsqueeze(0)), 0)
                    action_c1 = observed_c[act_idx1, batch_idx, :].squeeze(1)                
                    action_c2 = observed_c[act_idx2, batch_idx, :].squeeze(1)
                    action_c = (action_c1 + action_c2) / 2
              
            if self.cell == 'lstm':
                weighted_h = self.lamda * action_h + (1-self.lamda)*cur_h
                weighted_c = self.lamda * action_c + (1-self.lamda)*cur_c
                rnn_state = (weighted_h, weighted_c)
                cur_h, cur_c = self.rnn(cur_input, rnn_state)
            else:
                weighted_h = self.lamda * action_h + (1-self.lamda)*cur_h
                cur_h = self.rnn(cur_input, weighted_h)
              
            h.append(cur_h)
        
        demo = demo.unsqueeze(1).repeat(1, time_step, 1)
        demo = demo.contiguous().view(-1, self.demo_dim)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)
        rnn_reshape = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_reshape = self.nn_dropout(rnn_reshape)

        rnn_reshape = torch.cat((rnn_reshape, demo), dim=1)
        rnn_reshape = self.fusion(rnn_reshape)
        output = self.output(rnn_reshape)
        output = self.sigmoid(output)
        output = output.contiguous().view(rnn_outputs.size(0), -1, output.size(-1))

        return output

def get_loss(model, pred, true, mask, gamma=0.9, entropy_term=0.01, use_baseline=True, device=None):
    bce_loss = nn.BCELoss(reduction='none')
    y_onehot = torch.zeros_like(pred).to(device)
    y_onehot.scatter_(2, true, 1)
    loss_task = bce_loss(pred, y_onehot).sum(dim=2)
    loss_task = (loss_task * mask.squeeze(-1)).sum(dim=1) / mask.squeeze(-1).sum(dim=1)
    loss_task = loss_task.mean()
    
    act_prob1 = model.agent1_prob
    act_prob1 = torch.stack(act_prob1).permute(1, 0).to(device)
    act_prob1 = act_prob1 * mask.view(act_prob1.size(0), act_prob1.size(1)) 
    act_entropy1 = model.agent1_entropy
    act_entropy1 = torch.stack(act_entropy1).permute(1, 0).to(device)
    act_entropy1 = act_entropy1 * mask.view(act_entropy1.size(0), act_entropy1.size(1)) 
    if use_baseline == True:
        act_baseline1 = model.agent1_baseline
        act_baseline1 = torch.stack(act_baseline1).squeeze(-1).permute(1, 0).to(device)
        act_baseline1 = act_baseline1 * mask.view(act_baseline1.size(0), act_baseline1.size(1))
    
    act_prob2 = model.agent2_prob
    act_prob2 = torch.stack(act_prob2).permute(1, 0).to(device)
    act_prob2 = act_prob2 * mask.view(act_prob2.size(0), act_prob2.size(1)) 
    act_entropy2 = model.agent2_entropy
    act_entropy2 = torch.stack(act_entropy2).permute(1, 0).to(device)
    act_entropy2 = act_entropy2 * mask.view(act_entropy2.size(0), act_entropy2.size(1)) 
    if use_baseline == True:
        act_baseline2 = model.agent2_baseline
        act_baseline2 = torch.stack(act_baseline2).squeeze(-1).permute(1, 0).to(device)
        act_baseline2 = act_baseline2 * mask.view(act_baseline2.size(0), act_baseline2.size(1))

    rewards = (pred * y_onehot).sum(dim=-1)
    rewards = rewards * mask.squeeze(-1)
      
    running_rewards = []
    discounted_rewards = 0
    for i in reversed(range(len(rewards[0, :]))):
        discounted_rewards = rewards[:, i] + gamma * discounted_rewards
        running_rewards.insert(0, discounted_rewards)
    rewards = torch.stack(running_rewards).permute(1, 0)
    rewards = (rewards - rewards.mean(dim=1).unsqueeze(-1)) / (rewards.std(dim=1) + 1e-7).unsqueeze(-1)
    rewards = rewards.detach()
    
    if use_baseline == True:
        loss_value1 = torch.sum((rewards - act_baseline1) ** 2, dim=1) / torch.sum(mask, dim=1)
        loss_value1 = torch.mean(loss_value1)
        loss_value2 = torch.sum((rewards - act_baseline2) ** 2, dim=1) / torch.sum(mask, dim=1)
        loss_value2 = torch.mean(loss_value2)
        loss_value = loss_value1 + loss_value2
        loss_RL1 = -torch.sum(act_prob1 * (rewards - act_baseline1) + entropy_term * act_entropy1, dim=1) / torch.sum(mask, dim=1)
        loss_RL1 = torch.mean(loss_RL1)
        loss_RL2 = -torch.sum(act_prob2 * (rewards - act_baseline2) + entropy_term * act_entropy2, dim=1) / torch.sum(mask, dim=1)
        loss_RL2 = torch.mean(loss_RL2)
        loss_RL = loss_RL1 + loss_RL2
        loss = loss_RL + loss_task + loss_value
    else:
        loss_RL1 = -torch.sum(act_prob1 * rewards + entropy_term * act_entropy1, dim=1) / torch.sum(mask, dim=1)
        loss_RL1 = torch.mean(loss_RL1)
        loss_RL2 = -torch.sum(act_prob2 * rewards + entropy_term * act_entropy2, dim=1) / torch.sum(mask, dim=1)
        loss_RL2 = torch.mean(loss_RL2)
        loss_RL = loss_RL1 + loss_RL2
        loss = loss_RL + loss_task
    
    return loss, loss_RL, loss_task
