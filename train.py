import numpy as np
import torch
import math
import random
from torch.autograd import Variable
from itertools import count
from torch.optim.lr_scheduler import StepLR

def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

class Optimizer_AC(object):
    def __init__(self, actor, critic, q_optimizer, policy_optimizer, memory, loss_fn,
                q_scheduler = None):
        self.actor = actor
        self.critic = critic
        self.q_optimizer = q_optimizer
        self.policy_optimizer = policy_optimizer
        self.memory = memory
        self.loss_fn = loss_fn
        if q_scheduler == None:
            self.q_scheduler = StepLR(q_optimizer, step_size = 1, gamma = 1.0)
        else:
            self.q_scheduler = q_scheduler
        
    def step(self, transitions):
        self.q_scheduler.step()
        BATCH_SIZE = len(transitions)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.memory.Transition(*transitions)
        state_batch = to_var(batch.state)
        #state_batch.requires_grad_(True)

        reward_batch = to_var(batch.reward)
        action_batch = to_var(batch.action)
        #print(action_batch.requires_grad)
        #reward_batch.requires_grad_(False)

        #Unfreeze weights of Critic
        for param in self.critic.parameters():
            param.requires_grad = True

        q = self.critic(state_batch, action_batch.view(-1, 1))
        #print(q)
        #print(reward_batch)
        self.q_optimizer.zero_grad()
        q_loss = self.loss_fn(q, reward_batch)
        q_loss.backward()
        #print(q_loss)
        #print(q_loss.requires_grad)
        self.q_optimizer.step()
        #action_batch.requires_grad_(False)

        #Compute loss for actor
        self.policy_optimizer.zero_grad()

        #Freeze weights of Critic
        for param in self.critic.parameters():
            param.requires_grad = False

        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        #print(policy_loss.requires_grad)
        
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policy_optimizer.step()
        #print(q_loss.item()/BATCH_SIZE, policy_loss.item()/BATCH_SIZE)
        return q.mean().item(), q_loss.item()/BATCH_SIZE, policy_loss.item()/BATCH_SIZE

class Optimizer_Actor(object):
    def __init__(self, actor, critic, policy_optimizer, memory, loss_fn,
                q_scheduler = None):
        self.actor = actor
        self.critic = critic
        self.policy_optimizer = policy_optimizer
        self.memory = memory
        self.loss_fn = loss_fn
        if q_scheduler == None:
            self.q_scheduler = StepLR(q_optimizer, step_size = 1, gamma = 1.0)
        else:
            self.q_scheduler = q_scheduler
        
    def step(self, transitions):
        self.q_scheduler.step()
        BATCH_SIZE = len(transitions)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.memory.Transition(*transitions)
        state_batch = to_var(batch.state)
        #state_batch.requires_grad_(True)

        reward_batch = to_var(batch.reward)
        action_batch = to_var(batch.action)
        #print(action_batch.requires_grad)
        #reward_batch.requires_grad_(False)
        
        q = self.critic(state_batch, action_batch.view(-1, 1))
        #Freeze weights of Critic
        for param in self.critic.parameters():
            param.requires_grad = False

        #Compute loss for actor
        self.policy_optimizer.zero_grad()

        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        #print(policy_loss.requires_grad)
        
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policy_optimizer.step()
        #print(q_loss.item()/BATCH_SIZE, policy_loss.item()/BATCH_SIZE)
        return q.mean().item(), 0, policy_loss.item()/BATCH_SIZE

    
class Optimizer_Critic(object):
    def __init__(self, critic, q_optimizer, memory, loss_fn,
                q_scheduler = None):
        self.critic = critic
        self.q_optimizer = q_optimizer
        self.memory = memory
        self.loss_fn = loss_fn
        if q_scheduler == None:
            self.q_scheduler = StepLR(q_optimizer, step_size = 1, gamma = 1.0)
        else:
            self.q_scheduler = q_scheduler
        
    def step(self, transitions):
        BATCH_SIZE = len(transitions)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.memory.Transition(*transitions)
        reward_batch = to_var(batch.reward)
        state_batch = to_var(batch.state)
        action_batch = to_var(batch.action)

        #Unfreeze weights of Critic
        for param in self.critic.parameters():
            param.requires_grad = True

        q = self.critic(state_batch, action_batch.view(-1, 1))
        #print(q)
        #print(reward_batch)
        self.q_optimizer.zero_grad()
        q_loss = self.loss_fn(q, reward_batch)
        q_loss.backward()
        #print(q_loss)
        #print(q_loss.requires_grad)
        self.q_optimizer.step()
        return q.mean().item(), q_loss.item()/BATCH_SIZE, 0
    

class Artificial_environment_noise_t(object):
    def __init__(self, noise, env, noise_power = 1.0):
        self.noise = noise
        self.env = env
        self.noise_power = noise_power

    def step(self):
        with torch.no_grad():
            action = to_var(torch.from_numpy(self.noise_power*self.noise()).type(torch.FloatTensor))
            next_state, reward, done, _ = self.env.step(action.numpy())
            reward = to_var(torch.tensor([-1.0]).type(torch.FloatTensor))
            if next_state[0] >= self.goal_position():
                reward = to_var(torch.tensor([100.]).type(torch.FloatTensor))
                    
            return next_state, reward, done, action

    def reset(self):
        return self.env.reset()

    def goal_position(self):
        return self.env.goal_position
    
    def render(self):
        self.env.render()

class Artificial_environment_base_t(object):
    def __init__(self, actor, env):
        self.actor = actor
        self.env = env
        
    def step(self):
        with torch.no_grad():
            action = to_var(self.actor(torch.from_numpy(self.env.state).type(torch.FloatTensor)))
            next_state, reward, done, _ = self.env.step(action.numpy())
            reward = to_var(torch.tensor([reward]).type(torch.FloatTensor))

            return next_state, reward, done, action

    def reset(self):
        return self.env.reset()

    def goal_position(self):
        return self.env.goal_position
    
    def render(self):
        self.env.render()
    
class Artificial_environment_base_and_noise_t(object):
    def __init__(self, actor, noise, env, epsilon, epsilon_decay, noise_power = 1.0):
        self.actor = actor
        self.noise = noise
        self.env = env
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.noise_power = noise_power
        
    def step(self):
        with torch.no_grad():
            action = to_var(self.actor(torch.from_numpy(self.env.state).type(torch.FloatTensor)))
            self.epsilon -=  self.epsilon_decay
            action += to_var(torch.from_numpy(self.noise_power*self.noise()*max(0, self.epsilon)).type(torch.FloatTensor))
            next_state, reward, done, _ = self.env.step(action.detach().numpy())
            reward = to_var(torch.tensor([reward]).type(torch.FloatTensor))

            return next_state, reward, done, action

    def reset(self):
        return self.env.reset()
    
    def goal_position(self):
        return self.env.goal_position

    def render(self):
        self.env.render()

class Launcher_t(object):
    def __init__(self, Artificial_environment, memory = None, GAMMA = 0.999, BATCH_SIZE = 32):
        self.Artificial_environment = Artificial_environment
        self.memory = memory
        self.history_rewards = []
        self.history_discounted_rewards = []
        
        self.successes = 0
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.counter = 0
        
    def step(self, render = False):
        with torch.no_grad():
            state = to_var(torch.from_numpy(self.Artificial_environment.reset()).type(torch.FloatTensor))
            states = []
            rewards = []
            actions = []

        for t in count():
            with torch.no_grad():
                if render:
                    self.Artificial_environment.render()
                
                next_state, reward, done, action = self.Artificial_environment.step()
                
                if not done:
                    next_state = to_var(torch.from_numpy(next_state).type(torch.FloatTensor))
                    # Move to the next state
                    state = next_state
                else:
                    #print(next_state)
                    if next_state[0] >= self.Artificial_environment.goal_position():
                        self.successes += 1
                    next_state = None

                states.append(state.detach())
                rewards.append(reward.detach())
                actions.append(action.detach())

            if done:
                discounted_rewards = []
                discounted_rewards.append(0)
                for i in range(len(rewards)):
                    discounted_rewards[0] += rewards[i]*(self.GAMMA**(i))
                for i in range(1, len(rewards)):
                    discounted_rewards.append((discounted_rewards[i-1] - rewards[i-1])/self.GAMMA)
                if self.memory != None:
                    # Store the transitions in memory
                    for i in range(len(rewards)):
                        self.memory.push(states[i], actions[i], discounted_rewards[i]/20.)
                self.history_rewards.append(np.array(rewards).mean())
                self.history_discounted_rewards.append(np.array(discounted_rewards).mean()/20.)
                self.counter += 1
                
                break

    def history(self, begin = 0, end = None):
        return self.history_rewards[begin:end], self.history_discounted_rewards[begin:end], self.successes

def train_wo_launcher(memory, BATCH_SIZE, optimizer, num_episodes, output = True):
    history_loss = []
    for i_episode in range(num_episodes):
        loss = []
        if output:
            critic_predictions = []
        loader = torch.utils.data.DataLoader(
                 dataset=memory,
                 batch_size=BATCH_SIZE,
                 shuffle=True)
        for transitions in loader:
            #print(transitions)
            q, q_loss, p_loss = optimizer.step(transitions)
            loss.append((q_loss, p_loss))
            if output:
                critic_predictions.append(q)
        #print(loss)
        loss = [*zip(*loss)]
        history_loss.append((sum(loss[0])/len(loss[0]), sum(loss[1])/len(loss[1]))) 
        if output:
            print("episode ", i_episode, ": loss = ", history_loss[-1], " q = ", sum(critic_predictions)/len(critic_predictions))
    return history_loss


def train(Launcher, optimizer, num_episodes, render = False, output = True):
    history_loss = []
    for i_episode in range(num_episodes):
        Launcher.step(render)
        self.optimizer.q_scheduler.step()
        if output:
            reward, discounted_reward, successes = Launcher.history(-1, None)
            critic_predictions = []
        loss = []
        loader = torch.utils.data.DataLoader(
                 dataset=Launcher.memory,
                 batch_size=Launcher.BATCH_SIZE,
                 shuffle=True)
        for transitions in loader:
            #print(transitions)
            q, q_loss, p_loss = optimizer.step(transitions)
            loss.append((q_loss, p_loss))
            if output:
                critic_predictions.append(q)
        #print(loss)
        loss = [*zip(*loss)]
        history_loss.append((sum(loss[0])/len(loss[0]), sum(loss[1])/len(loss[1]))) 
        if output:
            print("episode ", i_episode, ": loss = ", history_loss[-1], " reward = ", reward, " discounted_reward = ", discounted_reward, " successes = ", successes, " q = ", sum(critic_predictions)/len(critic_predictions))
    history_rewards, _, successes = Launcher.history(-num_episodes, None)
    return history_loss, history_rewards, successes/num_episodes

def test(Launcher, num_episodes, render = False, output = False):
    for i_episode in range(num_episodes):
        Launcher.step(render)
        if output:
            reward, discounted_reward, successes = Launcher.history(-1, None)
            print("episode ", i_episode, "reward = ", reward, " discounted_reward = ", discounted_reward, " successes = ", successes)
    history_rewards, _, successes = Launcher.history(-num_episodes, None)
    return history_rewards, successes/num_episodes

def fill_memory(Launcher, num_episodes):
    for i_episode in range(num_episodes):
        Launcher.step()
    return Launcher.history(-num_episodes, None)