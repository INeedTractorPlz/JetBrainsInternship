from memory import Transition
import numpy as np
import torch
import math
import random

def select_action(model, env, state, steps_done, EPS_START, EPS_END, EPS_DECAY):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            mu_v, var_v, _ = model(state)
            mu = mu_v.data.cpu().numpy()
            sigma = torch.sqrt(var_v).data.cpu().numpy()
            action = np.random.normal(mu, sigma)
            return action 
    else:
        return np.array([float(random.randrange(env.min_action, env.max_action, 1))])
    #mu_v, var_v, _ = model(state)
    #mu = mu_v.data.cpu().numpy()
    #sigma = torch.sqrt(var_v).data.cpu().numpy()
    #action = np.random.normal(mu, sigma)
    #return action

def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2

def optimize_model(model, optimizer, memory, loss_fn, BATCH_SIZE, ENTROPY_BETA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    state_batch = torch.stack(batch.state, dim = 0)
    state_batch.requires_grad_(True)
    
    reward_batch = torch.cat(batch.reward)
    action_batch = torch.cat(batch.action)
    
    optimizer.zero_grad()
    mu_v, var_v, value_v = model(state_batch)
    #print(mu_v.requires_grad, var_v.requires_grad, value_v.requires_grad)
    
    #loss_value_v = loss_fn(value_v, reward_batch) //- value_v.detach()
    log_prob_v = (reward_batch)* calc_logprob(mu_v, var_v, action_batch)
    loss_policy_v = -log_prob_v.mean()
    #entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

    loss_v = loss_policy_v 
    #+ entropy_loss_v + loss_value_v
    loss_v.backward()
    optimizer.step()
    
def optimize_model_AC(actor, critic, q_optimizer, policy_optimizer, memory, loss_fn, BATCH_SIZE):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    state_batch = torch.stack(batch.state, dim = 0)
    #state_batch.requires_grad_(True)
    
    reward_batch = torch.cat(batch.reward)
    action_batch = torch.cat(batch.action)
    #print(action_batch.requires_grad)
    #reward_batch.requires_grad_(False)
    
    #Unfreeze weights of Critic
    for param in critic.parameters():
        param.requires_grad = True
    
    q = critic(state_batch, action_batch.view(-1, 1))
           
    q_optimizer.zero_grad()
    q_loss = loss_fn(q, reward_batch) 
    #print(q_loss.requires_grad)
    q_optimizer.step()
    #action_batch.requires_grad_(False)
    
    #Compute loss for actor
    policy_optimizer.zero_grad()
    
    #Freeze weights of Critic
    for param in critic.parameters():
        param.requires_grad = False
    
    policy_loss = -critic(state_batch, actor(state_batch))
    #print(policy_loss.requires_grad)
    
    
    policy_loss = policy_loss.mean()
    policy_loss.backward()
    policy_optimizer.step()