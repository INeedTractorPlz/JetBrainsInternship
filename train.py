import numpy as np
import torch
import math
import random
from torch.autograd import Variable
from itertools import count
from torch.optim.lr_scheduler import StepLR

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

class ModelQ_t:
    def __init__(self, policy_net, target_net, env,  GAMMA, steps_done = 1, EPS_END = 0, EPS_START = 0, EPS_DECAY = 1, memory = None, BATCH_SIZE = None, loss_fn = None, optimizer = None, TARGET_UPDATE = None, factor = 1):
        self.policy_net = policy_net
        self.target_net = target_net
        self.env = env
        self.steps_done = steps_done
        self.EPS_DECAY = EPS_DECAY
        self.EPS_END = EPS_END
        self.EPS_START = EPS_START
        self.memory = memory
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.TARGET_UPDATE = TARGET_UPDATE
        self.factor = factor
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                Q = self.policy_net(state)
                action = torch.argmax(Q)
                return torch.tensor(action - 1.).type(torch.FloatTensor)*self.factor
        else:
            return torch.tensor(random.randrange(3) - 1.).type(torch.FloatTensor)*self.factor
    
    def reward_function(self, next_state):
        if next_state[0] >= self.env.goal_position:
            reward = 100.
        else:
            reward = -1.
        #reward = -1.
        return torch.tensor([reward]).type(torch.FloatTensor)
    
    def optimize_model(self, transitions):
        #if len(self.memory) < self.BATCH_SIZE:
        #    return
        #transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        #batch = self.memory.Transition(*zip(*transitions))
        batch = self.memory.Transition(*transitions)
        BATCH_SIZE = batch.state.shape[0]
        #print(transitions)
        #print(batch.state.shape)
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        
        #non_final_mask_1 = flatten(batch.next_s1 != 500)
        #non_final_next_states_1 = torch.stack([s for s in batch.next_s1
        #                                            if s is not 500], dim = 0)
        #non_final_mask_2 = flatten(batch.next_s1 != 500)
        #non_final_next_states_1 = torch.stack([s for s in batch.next_s1
        #                                            if s is not 500], dim = 0)
        #non_final_mask_3 = flatten(batch.next_s3 != 500)
        #non_final_next_states_1 = torch.stack([s for s in batch.next_s1
        #                                            if s is not 500], dim = 0)
        
        non_final_mask_1 = torch.tensor(tuple(map(lambda s: s[0] != 500.,
                                             batch.next_s1)), dtype=torch.uint8)
        non_final_next_states_1 = torch.stack([s for s in batch.next_s1
                                                    if s[0] != 500.], dim = 0)
        non_final_mask_2 = torch.tensor(tuple(map(lambda s: s[0] != 500.,
                                              batch.next_s2)), dtype=torch.uint8)
        non_final_next_states_2 = torch.stack([s for s in batch.next_s2
                                                    if s[0] != 500.], dim = 0)
        non_final_mask_3 = torch.tensor(tuple(map(lambda s: s[0] != 500.,
                                              batch.next_s3)), dtype=torch.uint8)
        non_final_next_states_3 = torch.stack([s for s in batch.next_s3
                                                    if s[0] != 500.], dim = 0)
        
        #for s in batch.next_s3:
        #    if s[0] == 500.:
        #        print(batch.next_s3)
        #state_batch = torch.stack(batch.state, dim = 0)
        #reward_batch_1 = torch.cat(batch.reward_1)
        #reward_batch_2 = torch.cat(batch.reward_2)
        #reward_batch_3 = torch.cat(batch.reward_3)
        
        state_batch = batch.state
        reward_batch_1 = batch.reward_1
        reward_batch_2 = batch.reward_2
        reward_batch_3 = batch.reward_3

        state_batch.requires_grad_(True)
        prediction = self.policy_net(state_batch)

        next_state_values_1 = torch.zeros(BATCH_SIZE)
        next_state_values_1[non_final_mask_1] = self.target_net(non_final_next_states_1).max(1)[0].detach()

        next_state_values_2 = torch.zeros(BATCH_SIZE)
        next_state_values_2[non_final_mask_2] = self.target_net(non_final_next_states_2).max(1)[0].detach()

        next_state_values_3 = torch.zeros(BATCH_SIZE)
        next_state_values_3[non_final_mask_3] = self.target_net(non_final_next_states_3).max(1)[0].detach()

        next_state_values = torch.stack([next_state_values_1, next_state_values_2,
                                         next_state_values_3], dim = 1)
        #print(reward_batch_3.shape)
        reward_batch = torch.stack([flatten(reward_batch_1), flatten(reward_batch_2),
                                         flatten(reward_batch_3)], dim = 1)
        #print(next_state_values.shape, reward_batch.shape, prediction.shape)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        #print(sum([s is None for s in expected_state_action_values]))
        loss = self.loss_fn(expected_state_action_values, prediction)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, num_episodes, amplification_success = 1, render = False):
        self.steps_done = 0
        history_loss = []
        history_reward = []
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            state = torch.from_numpy(self.env.reset()).type(torch.FloatTensor)
            rewards = []
            for t in count():
                if render:
                    self.env.render()
                next_state_1, reward_1, done_1, _ = self.env.step([-1.*self.factor])
                
                reward_1 = self.reward_function(next_state_1)
                
                if not done_1:
                    next_state_1 = torch.from_numpy(next_state_1).type(torch.FloatTensor)
                else:
                    next_state_1 = torch.tensor([500., 500.]).type(torch.FloatTensor)
                    #print(t, " Nan1")
                
                self.env.state = np.array([state[0], state[1]])

                next_state_2, reward_2, done_2, _ = self.env.step([0.*self.factor])
                
                reward_2 = self.reward_function(next_state_2)
                
                if not done_2:
                    next_state_2 = torch.from_numpy(next_state_2).type(torch.FloatTensor)
                else:
                    next_state_2 = torch.tensor([500., 500.]).type(torch.FloatTensor)
                    #print(t, " Nan2")
                
                self.env.state = np.array([state[0], state[1]])

                next_state_3, reward_3, done_3, _ = self.env.step([1.*self.factor])
                
                reward_3 = self.reward_function(next_state_3)
                
                if not done_3:
                    next_state_3 = torch.from_numpy(next_state_3).type(torch.FloatTensor)
                else:
                    next_state_3 = torch.tensor([500., 500.]).type(torch.FloatTensor)
                    #print(t, " Nan3")
                    
                self.env.state = np.array([state[0], state[1]])

                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step([action.item()])
                reward = self.reward_function(next_state)
                
                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                
                #print(state, next_state_1, next_state_2, next_state_3, 
                #            reward_1, reward_2, reward_3)
                # Store the transition in memory
                if reward_1 + reward_2 + reward_3 > 0:
                    print("nearly")
                    amplification = amplification_success
                else:
                    amplification = 1
                for i in range(amplification):
                    self.memory.push(state, next_state_1, next_state_2, next_state_3, 
                                reward_1, reward_2, reward_3)
                
                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                #history_loss.append(self.optimize_model())
                rewards.append(reward)
                if done:
                    print(next_state, reward_1, reward_2, reward_3)
                    if next_state[0] >= self.env.goal_position:
                        print("success!!")
                    history_reward.append(sum(rewards)/len(rewards))
                    break
            
            loader = torch.utils.data.DataLoader(
                     dataset=self.memory,
                     batch_size=self.BATCH_SIZE,
                     shuffle=True)
            losses = []
            for transitions in loader:
                #print(transitions)
                losses.append(self.optimize_model(transitions))
            history_loss.append(sum(losses)/len(losses))

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')
        self.env.close()
        return history_loss, history_reward
        
    def test(self, num_episodes, render = True, memory = None, sequence = False):
        successes = 0
        history_rewards = []
        history_discounted_rewards = []
        
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            state = torch.from_numpy(self.env.reset()).type(torch.FloatTensor)
            if memory != None:     
                rewards = []
                actions = []
                states = []
            
            for t in count():
                # Select and perform an action
                if render:
                    self.env.render()
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step([action.item()])
                action = torch.tensor([action]).type(torch.FloatTensor)
                
                if next_state[0] >= self.env.goal_position:
                        reward = torch.tensor([100.]).type(torch.FloatTensor)
                else:
                    reward = torch.tensor([-1.]).type(torch.FloatTensor)
                
                if memory != None:
                    rewards.append(reward)
                    actions.append(action)
                    states.append(state)

                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                state = next_state

                if done:
                    if memory != None:
                        discounted_rewards = []
                        discounted_rewards.append(0)
                        for i in range(len(rewards)):
                            discounted_rewards[0] += rewards[i]*(self.GAMMA**(i))/20.
                        for i in range(1, len(rewards)):
                            discounted_rewards.append((discounted_rewards[i-1] - rewards[i-1]/20.)/self.GAMMA)
                        # Store the transitions in memory
                        if sequence:
                            memory.push(states, actions, discounted_rewards)
                        else:
                            for i in range(len(rewards)):
                                memory.push(states[i], actions[i], discounted_rewards[i])
                        history_rewards.append(np.array(rewards).mean())
                        history_discounted_rewards.append(np.array(discounted_rewards).mean())
                
                    print(state, reward)
                    if state[0] >= self.env.goal_position:
                        successes += 1
                    break
        print(successes/num_episodes*100)
        print('Complete')
        self.env.close()
        if memory != None:
            return history_rewards, history_discounted_rewards


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

def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2

class Optimizer_RNN(object):
    def __init__(self, rnn, optimizer, memory, loss_fn,
                scheduler = None):
        self.rnn = rnn
        self.optimizer = optimizer
        self.memory = memory
        self.loss_fn = loss_fn
        if scheduler == None:
            self.scheduler = StepLR(optimizer, step_size = 1, gamma = 1.0)
        else:
            self.scheduler = scheduler
        
    def step(self, sequences):
        BATCH_SIZE = len(sequences)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        #batch = self.memory.Transition(*zip(*sequences))
        batch = self.memory.Transition(*map(lambda y: torch.transpose(torch.stack(list(map(lambda x: torch.stack(x),y))),0 , 1), 
              list(zip(*sequences))))
        reward_batch = to_var(batch.reward)
        state_batch = to_var(batch.state)
        action_batch = to_var(batch.action)

        loss_rewards = []
        loss_states = []
        rewards = []
        len_seq = len(sequences[0])
        self.optimizer.zero_grad()
        h,c = self.rnn.initial_state(BATCH_SIZE)
        for i in range(len_seq):
            #print(action_batch.shape, state_batch.shape, h.shape)
            output, (h,c) = self.rnn(torch.cat([action_batch[i], state_batch[i], h[-1]], dim = 1).view(1, BATCH_SIZE, self.rnn.size_input), (h, c))
            #print(output.shape)
            loss_reward = self.loss_fn(output[:, 0], reward_batch[i][:, 0])
            loss_state = calc_logprob(h[-1][:, 0], torch.abs(h[-1][:, 1]), state_batch[i][:, 0]) + calc_logprob(h[-1][:, 2], torch.abs(h[-1][:, 3]), state_batch[i][:, 1])
            #print(loss_reward, loss_state)
            loss_reward.backward(retain_graph=True)
            loss_state.backward(retain_graph=True)
            loss_rewards.append(loss_reward.mean().item())
            loss_states.append(loss_state.mean().item())
            rewards.append(h[-1][:, 0].mean().item())
            
        self.optimizer.step()
        
        return sum(loss_rewards)/len_seq, sum(loss_states)/len_seq, sum(rewards)/len_seq 


class Trainer_Actor_RNN(object):
    def __init__(self, actor, rnn, env, loss_fn, optimizer_actor, GAMMA, render= True,
                optimizer_rnn = None, scheduler_actor = None, scheduler_rnn = None,
                artificial_reward = True):
        self.rnn = rnn
        self.optimizer_rnn = optimizer_rnn
        self.optimizer_actor = optimizer_actor
        self.loss_fn = loss_fn
        self.actor = actor
        self.env = env
        self.render = render
        self.GAMMA = GAMMA
        self.artificial_reward =artificial_reward
        if scheduler_actor == None:
            self.scheduler_actor = StepLR(optimizer_actor, step_size = 1, gamma = 1.0)
        else:
            self.scheduler_actor = scheduler_actor
        if scheduler_rnn == None:
            self.scheduler_rnn = StepLR(optimizer_rnn, step_size = 1, gamma = 1.0)
        else:
            self.scheduler_rnn = scheduler_rnn
    
    def reward_function(self, next_state):
        if next_state[0] >= self.env.goal_position:
            reward = 100.
        else:
            reward = -1.
        #reward = -1.
        return torch.tensor([reward]).type(torch.FloatTensor)
    
    def train(self, num_episodes, train_rnn = True, BATCH_SIZE = 1):
        losses_rnn = []
        losses_actor = []
        for episode in range(num_episodes):
            loss_rnn = torch.tensor([0]).type(torch.FloatTensor)
            loss_actor = torch.tensor([0]).type(torch.FloatTensor)
            len_seq = 0
            rewards = []
            outputs = []
            if train_rnn:
                self.optimizer_rnn.zero_grad()
            self.optimizer_actor.zero_grad()
            h, c = self.rnn.initial_state(BATCH_SIZE)
            state = to_var(torch.from_numpy(self.env.reset()).type(torch.FloatTensor))
            done = False
            while not done:
                len_seq += 1
                if self.render:
                    self.env.render()
                action = to_var(self.actor(torch.cat([state,h[-1][0]],0).type(torch.FloatTensor)))
                next_state, reward, done, _ = self.env.step(action.numpy())
                if self.artificial_reward:
                    reward = self.reward_function(next_state)
                else:
                    reward = torch.tensor([reward]).type(torch.FloatTensor)
                if next_state[0] >= self.env.goal_position:
                    print("success!!")
                output, (h, c) = self.rnn(torch.cat([action, state, h[-1][0]], 0).view(1, BATCH_SIZE, self.rnn.size_input), (h, c))
                rewards.append(reward)
                outputs.append(output[0, :, 0])
                loss_actor -= output[0, :, 0]
                state = to_var(torch.from_numpy(next_state).type(torch.FloatTensor))
            
            discounted_rewards = []
            discounted_rewards.append(torch.tensor([0]).type(torch.FloatTensor))
            for i in range(len_seq):
                discounted_rewards[0] += rewards[i]*(self.GAMMA**(i))/20.
            for i in range(1, len_seq):
                discounted_rewards.append((discounted_rewards[i-1] - rewards[i-1]/20.)/self.GAMMA)
            
            for i in range(len_seq):
                #print(outputs[i], discounted_rewards[i])
                loss_rnn += self.loss_fn(outputs[i], discounted_rewards[i])
                
            loss_actor = loss_actor/len_seq
            loss_rnn/len_seq
            
            
            #Freeze weights of RNN
            for param in self.rnn.parameters():
                param.requires_grad = False
            
            if train_rnn:
                #And unfreeze weights of actor
                for param in self.actor.parameters():
                    param.requires_grad = True
            
            loss_actor.backward(retain_graph=train_rnn)
            self.optimizer_actor.step()                                          
            
            
            #Unfreeze weights of RNN
            for param in self.rnn.parameters():
                param.requires_grad = True
            
            if train_rnn:     
                #And freeze weights of actor
                for param in self.actor.parameters():
                    param.requires_grad = False
                loss_rnn.backward(retain_graph=train_rnn)
                self.optimizer_rnn.step()                                      
            print("episode = ", episode, " loss_rnn = ", loss_rnn, " loss_actor", loss_actor)
                                                      
            losses_rnn.append(loss_rnn.item())                                         
            losses_actor.append(loss_actor.item())                                         
            if train_rnn:
                self.scheduler_rnn.step()
            self.scheduler_actor.step()
                                                      
        return sum(losses_actor)/num_episodes, sum(losses_rnn)/num_episodes 

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

        
class Artificial_environment_rnn_t(object):
    def __init__(self, rnn, actor, env):
        self.rnn = rnn
        self.actor = actor
        self.env = env
        self.h, self.c = rnn.initial_state(1)
    def step(self):
        with torch.no_grad():
            state = torch.from_numpy(self.env.state).type(torch.FloatTensor)
            action = to_var(self.actor(torch.cat([state,self.h[-1][0]],0).type(torch.FloatTensor)))
            next_state, reward, done, _ = self.env.step(action.numpy())
            reward = to_var(torch.tensor([reward]).type(torch.FloatTensor))
            output, (self.h, self.c) = rnn(torch.cat([action, state, self.h[-1][0]], 0).view(1, 1, self.rnn.size_input), (self.h, self.c)) 
            return next_state, reward, done, action

    def reset(self):
        
        return self.env.reset()

    def goal_position(self):
        return self.env.goal_position
    
    def render(self):
        self.env.render()

        
class Launcher_t(object):
    def __init__(self, Artificial_environment, memory = None, GAMMA = 0.999, 
                amplification_success = 1, sequence = False):
        self.Artificial_environment = Artificial_environment
        self.memory = memory
        self.history_rewards = []
        self.history_discounted_rewards = []
        
        self.successes = 0
        self.GAMMA = GAMMA
        self.counter = 0
        self.sequence = sequence
        self.amplification_success = amplification_success
        
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
                        success = True
                    else:
                        success = False
                    next_state = None

                states.append(state.detach())
                rewards.append(reward.detach())
                actions.append(action.detach())

            if done:
                discounted_rewards = []
                discounted_rewards.append(0)
                for i in range(len(rewards)):
                    discounted_rewards[0] += rewards[i]*(self.GAMMA**(i))/20.
                for i in range(1, len(rewards)):
                    discounted_rewards.append((discounted_rewards[i-1] - rewards[i-1]/20.)/self.GAMMA)
                if self.memory != None:
                    # Store the transitions in memory
                    if success:
                        amplification = self.amplification_success
                    else:
                        amplification = 1
                    for i in range(amplification):
                        if self.sequence:
                            self.memory.push(states, actions, discounted_rewards)
                        else:
                            for i in range(len(rewards)):
                                self.memory.push(states[i], actions[i], discounted_rewards[i])
                self.history_rewards.append(np.array(rewards).mean())
                self.history_discounted_rewards.append(np.array(discounted_rewards).mean())
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

def train_sequences_wo_launcher(data, BATCH_SIZE, optimizer, num_episodes, output = True):
    history_loss = []
    for i_episode in range(num_episodes):
        loss = []
        if output:
            rnn_predictions = []
        loader = torch.utils.data.DataLoader(
                 dataset=list(range(len(data))),
                 batch_size=BATCH_SIZE,
                 shuffle=True)
        for index in loader:
            #print(transitions)
            loss_r, loss_s, reward = optimizer.step(data.memory[index.numpy()])
            loss.append((loss_r, loss_s))
            if output:
                rnn_predictions.append(reward)
        #print(loss)
        loss = [*zip(*loss)]
        history_loss.append((sum(loss[0])/len(loss[0]), sum(loss[1])/len(loss[1]))) 
        if output:
            print("episode ", i_episode, ": loss = ", history_loss[-1], " reward predict = ", sum(rnn_predictions)/len(rnn_predictions))
        optimizer.scheduler.step()
        
    return history_loss

def train(Launcher, optimizer, num_episodes, render = False, output = True):
    history_loss = []
    for i_episode in range(num_episodes):
        Launcher.step(render)
        optimizer.q_scheduler.step()
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