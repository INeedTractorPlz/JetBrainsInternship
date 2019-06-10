import torch
import math
import random
from itertools import count
import numpy as np

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

class ModelQ_t:
    def __init__(self, policy_net, target_net, env, steps_done, EPS_END, EPS_START, EPS_DECAY, 
                memory, BATCH_SIZE, GAMMA, loss_fn, optimizer, TARGET_UPDATE):
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
                _, action = torch.max(Q, -1)
                return torch.tensor(action)
        else:
            return torch.tensor(random.randrange(self.env.action_space.n))

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
        
        non_final_mask_1 = torch.tensor(tuple(map(lambda s: s is not None,
                                             batch.next_s1)), dtype=torch.uint8)
        non_final_next_states_1 = torch.stack([s for s in batch.next_s1
                                                    if s is not None], dim = 0)
        non_final_mask_2 = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_s2)), dtype=torch.uint8)
        non_final_next_states_2 = torch.stack([s for s in batch.next_s2
                                                    if s is not None], dim = 0)
        non_final_mask_3 = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_s3)), dtype=torch.uint8)
        non_final_next_states_3 = torch.stack([s for s in batch.next_s3
                                                    if s is not None], dim = 0)

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

    def train(self, num_episodes, render = False):
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
                next_state_1, reward_1, done_1, _ = self.env.step(0)
                #reward_1 = next_state_1[0] + 0.5
                # Adjust reward for task completion
                #if next_state_1[0] >= 0.5:
                #    reward_1 += 1

                reward_1 = torch.tensor([reward_1])
                if not done_1:
                    next_state_1 = torch.from_numpy(next_state_1).type(torch.FloatTensor)
                else:
                    next_state_1 = None
                self.env.state = (state[0], state[1])

                next_state_2, reward_2, done_2, _ = self.env.step(1)
                #reward_2 = next_state_2[0] + 0.5
                # Adjust reward for task completion
                #if next_state_2[0] >= 0.5:
                #    reward_2 += 1

                reward_2 = torch.tensor([reward_2])
                if not done_2:
                    next_state_2 = torch.from_numpy(next_state_2).type(torch.FloatTensor)
                else:
                    next_state_2 = None
                self.env.state = (state[0], state[1])

                next_state_3, reward_3, done_3, _ = self.env.step(2)
                #reward_3 = next_state_3[0] + 0.5
                # Adjust reward for task completion
                #if next_state_3[0] >= 0.5:
                #    reward_3 += 1

                reward_3 = torch.tensor([reward_3])
                if not done_3:
                    next_state_3 = torch.from_numpy(next_state_3).type(torch.FloatTensor)
                else:
                    next_state_3 = None
                self.env.state = (state[0], state[1])

                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward])

                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                
                # Store the transition in memory
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
                losses.append(self.optimize_model(transitions))
            history_loss.append(sum(losses)/len(losses))
            
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')
        self.env.close()
        return history_loss, history_reward
        
    def test(self, num_episodes, render = True, memory = None):
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
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward]).type(torch.FloatTensor)
                action = torch.tensor([action]).type(torch.FloatTensor)
                
                if next_state[0] >= self.env.goal_position:
                        reward = torch.tensor([100.]).type(torch.FloatTensor)
                
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
                            discounted_rewards[0] += rewards[i]*(self.GAMMA**(i))
                        for i in range(1, len(rewards)):
                            discounted_rewards.append((discounted_rewards[i-1] - rewards[i-1])/self.GAMMA)
                        # Store the transitions in memory
                        for i in range(len(rewards)):
                            memory.push(states[i], actions[i], discounted_rewards[i]/20.)
                        history_rewards.append(np.array(rewards).mean())
                        history_discounted_rewards.append(np.array(discounted_rewards).mean()/20.)
                
                    print(state, reward)
                    if state[0] >= self.env.goal_position:
                        successes += 1
                    break
        print(successes/num_episodes*100)
        print('Complete')
        self.env.close()
        if memory != None:
            return history_rewards, history_discounted_rewards
        