import torch
import torch.nn as nn
import numpy as np

H1=200   #neurons of 1st layers
H2=200   #neurons of 2nd layers

def fanin_(size):
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 1)

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=0.003):
        super(Policy, self).__init__()
        self.state_space = state_dim
        self.action_space = action_dim
        #self.hidden = 200
        
        self.linear1 = nn.Linear(self.state_space, h1)
        #self.linear1.weight.data = fanin_(self.linear1.weight.data.size())

        self.linear2 = nn.Linear(h1, self.action_space)
        #self.linear2.weight.data = fanin_(self.linear2.weight.data.size())

        self.linear3 = nn.Linear(h2, self.action_space)
        #self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()

        #self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        #self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)
    
    def forward(self, x):    
        model = torch.nn.Sequential(
            self.linear1,
            #self.relu,
            self.linear2,
            #self.relu,
            #self.linear3
            #self.l1,
            #self.l2,
        )
        return model(x)

class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()
        self.hidden = 200
        self.base = nn.Sequential(
            nn.Linear(obs_size, self.hidden),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(self.hidden, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(self.hidden, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(self.hidden, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)

class RNN(nn.Module):
    def __init__(self, num_layers, size_hidden, size_input, size_output = 1):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.size_hidden = size_hidden
        self.size_input = size_input
        self.size_output = size_output
        self.lstm = nn.LSTM(self.size_input, self.size_hidden, self.num_layers)
        self.linear = nn.Linear(self.size_hidden, self.size_output)
        
    def forward(self, i, h):
        output_1, output_2 = self.lstm(i, h)
        return self.linear(output_1), output_2
    def initial_state(self, batch_size):
        h = torch.randn(self.num_layers, batch_size, self.size_hidden)
        c = torch.randn(self.num_layers, batch_size, self.size_hidden)
        return h, c

class Actor_RNN(nn.Module):
    def __init__(self, size_input, size_hidden, size_output = 1):
        super(Actor_RNN, self).__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.size_hidden = size_hidden
        self.linear_1 = nn.Linear(self.size_input, self.size_hidden)
        self.linear_2 = nn.Linear(self.size_hidden, self.size_output)
        
    def forward(self, input):
        return self.linear_2(self.linear_1(input))
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=3e-3):
        super(Critic, self).__init__()
                
        self.linear1 = nn.Linear(state_dim, h1)
        #self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
               
        self.linear2 = nn.Linear(h1+action_dim, h2)
        #self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
                
        self.linear3 = nn.Linear(h2, 1)
        #self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        #print(action.shape)
        x = self.linear1(state)
        #print(x.shape)
        x = self.relu(x)
        x = self.linear2(torch.cat([x,action],1))
        
        x = self.relu(x)
        x = self.linear3(x)
        
        return x

    
class Actor(nn.Module): 
    def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=0.003, power = 1.0):
        super(Actor, self).__init__()        
        self.linear1 = nn.Linear(state_dim, h1)
        #self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        
        
        self.linear2 = nn.Linear(h1, 1)
        #self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
                
        self.linear3 = nn.Linear(h2, action_dim)
        #self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.power = power
        
    def forward(self, state):
        x = self.linear1(state)
        #x = self.relu(x)
        x = self.linear2(x)
        #x = self.relu(x)
        #x = self.linear3(x)
        x = self.tanh(x)
        return self.power*x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]

class RandomActionNoise:
    def __init__(self, begin = 0, end = 1, shape = (1,)):
        self.begin = begin
        self.end = end
        self.shape = shape

    def __call__(self):
        return np.random.uniform(self.begin, self.end, self.shape)
    
class NormalActionNoise:
    def __init__(self, mu = 0, sigma = 1, size = 1):
        self.mu = mu
        self.sigma = sigma
        self.size = size
        
    def __call__(self):
        return np.random.normal(self.mu, self.sigma, size = self.size)

class BimodalActionNoise:
    def __init__(self, mu_1 = -1, sigma_1 = 1, mu_2 = 1, sigma_2 = 1, size = 1, clip = False):
        self.mu_1 = mu_1
        self.sigma_1 = sigma_1
        self.mu_2 = mu_2
        self.sigma_2 = sigma_2
        self.clip = clip
        self.size = size
        
    def __call__(self):
        a = (np.random.normal(self.mu_1, self.sigma_1, size = self.size) + np.random.normal(self.mu_2, self.sigma_2, size = self.size))/2.
        if self.clip:
            return np.clip(a, self.mu_1, self.mu_2)
        else:
            return a

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

#class Actor(nn.Module):
#    def __init__(self, obs_size, act_size):
#        super(Actor, self).__init__()
#        self.hidden = 200
#        self.base = nn.Sequential(
#            nn.Linear(obs_size, self.hidden),
#            nn.ReLU(),
#        )
#        self.mu = nn.Sequential(
#            nn.Linear(self.hidden, act_size),
#            nn.Tanh(),
#        )
#        self.var = nn.Sequential(
#            nn.Linear(self.hidden, act_size),
#            nn.Softplus(),
#        )
#        
#    def forward(self, x):
#        base_out = self.base(x)
#        return self.mu(base_out), self.var(base_out)