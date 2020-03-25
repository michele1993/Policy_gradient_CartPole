import torch
import gym
import torch.optim as opt
import numpy as np


from torch.distributions import Categorical


from Policy_network import Policy_net
from Baseline_net import Baseline_nn

n_episodes= 10000
discount = 0.99
max_t_step = 200
learning_rate = 1e-4
batch_size = 1

env = gym.make("CartPole-v1")
pol_nn = Policy_net().double()
base_nn = Baseline_nn().double()

criterion = torch.nn.MSELoss()

params = list(pol_nn.parameters()) + list(base_nn.parameters())

optimiser = opt.Adam(params,learning_rate)

episode_overall_return = []


for i in range(n_episodes):

    current_st = env.reset()
    episode_rwd = np.empty(0)
    episode_v_value = []
    episode_lp_action = []
    episode_states = []

    t = 0

    for t in range(max_t_step): #max_t_step

        mean_action = pol_nn(torch.tensor(current_st))

        d = Categorical(mean_action) # try to replace with bernulli and single output

        action = d.sample()

        episode_lp_action.append(d.log_prob(action))


        next_st,rwd, done, _ = env.step(action.numpy())

        predicted_value = base_nn(torch.tensor(current_st))

        episode_rwd = episode_rwd * discount

        episode_rwd = np.concatenate((episode_rwd,np.array([rwd])))



        episode_v_value.append(predicted_value)

        episode_states.append(current_st)



        if done:
            break

        current_st = next_st



    n_steps = 0

    policy_c = 0

    baseline_c = 0



    graph = True

    episode_rwd = np.flip(np.cumsum(np.flip(episode_rwd)))




    # perform update for each time step
    for e in range(t+1): # t episode_action_taken


        advantage = episode_rwd[e] - episode_v_value[e] # v_value


        # Update policy net

        policy_c += pol_nn.REINFORCE(episode_lp_action[e],advantage)


        baseline_c += torch.pow(episode_rwd[e] - episode_v_value[e], 2)

        n_steps += 1

        loss = pol_nn.REINFORCE(episode_lp_action[e],advantage) + torch.pow(episode_rwd[e] - episode_v_value[e], 2)

        optimiser.zero_grad()

        if e == t:
            graph = False

        loss.backward(retain_graph = graph)

        optimiser.step()

    episode_overall_return.append(n_steps)


    if i % 100 == 0:

        print("Baseline loss {}, Policy cost {}, Return {}, Episode {}".format(baseline_c[0]/n_steps,policy_c[0]/n_steps,sum(episode_overall_return)/100, i))

        episode_overall_return = []
