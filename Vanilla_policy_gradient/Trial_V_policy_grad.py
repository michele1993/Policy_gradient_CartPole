# By summing each time-step cost and then performing one update for the episode the policy still learns, but it's uch slower in
# learning as i should be, since the gradient of a sum should be equal to the sum of the gradients, but I guess by updating the baseline
# every time-step, you get more updated advantages when computing the policy gradient for each time-step, thus reducing the variance ?

import torch
import gym
import torch.optim as opt
import numpy as np


from torch.distributions import Categorical
#from torch.distributions import Bernoulli


from Vanilla_policy_gradient.Policy_network import Policy_net
from Vanilla_policy_gradient.Baseline_net import Baseline_nn

n_episodes= 10000
discount = 0.99
max_t_step = 200
learning_rate = 1e-3
batch_size = 1

env = gym.make("CartPole-v1")
pol_nn = Policy_net(output_size =2).double()
base_nn = Baseline_nn().double()

criterion = torch.nn.MSELoss()

params = list(pol_nn.parameters()) + list(base_nn.parameters())

optimiser = opt.Adam(params,learning_rate)

episode_overall_return = []


for i in range(n_episodes):

    current_st = env.reset()
    episode_rwd = torch.empty(0)
    episode_lp_action = torch.empty(0).float() #[]
    episode_states = np.empty(0)

    t = 0



    for t in range(max_t_step): #max_t_step

        episode_states= np.concatenate((episode_states,current_st),axis=0)

        mean_action = pol_nn(torch.tensor(current_st))

        d = Categorical(mean_action) # try to replace with bernulli and single output

        action = d.sample()

        episode_lp_action = torch.cat([episode_lp_action,torch.unsqueeze(d.log_prob(action).float(),dim=-1)])

        next_st, rwd, done, _ = env.step(int(action.numpy()))

        episode_rwd = episode_rwd * discount

        episode_rwd = torch.cat((episode_rwd,torch.tensor([rwd])),dim=-1)


        if done:
            break

        current_st = next_st



    predicted_value = base_nn(torch.tensor(episode_states.reshape(-1,4)))


    #episode_rwd = np.flip(np.cumsum(np.flip(episode_rwd)))

    episode_rwd = torch.flip(torch.cumsum(torch.flip(episode_rwd, (0,)), 0), (0,))


    advantage = episode_rwd.view(-1) - predicted_value.view(-1) # v_value


    # Update policy net

    policy_c = sum(pol_nn.REINFORCE(episode_lp_action,advantage))

    baseline_c = sum(torch.pow(advantage, 2))

    loss =  policy_c + baseline_c  #pol_nn.REINFORCE(episode_lp_action[e],advantage) + torch.pow(episode_rwd[e] - episode_v_value[e], 2)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    episode_overall_return.append(t)


    if i % 100 == 0:

        print("Baseline loss {}, Policy cost {}, Return {}, Episode {}".format(baseline_c.data/t,policy_c.data/t,sum(episode_overall_return)/100, i))

        episode_overall_return = []