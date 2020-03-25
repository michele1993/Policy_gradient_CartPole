import torch
import gym
import torch.optim as opt

from torch.distributions import Categorical

from AC_algorithm.Actor_NN import Actor_net
from AC_algorithm.Critic_NN import Critic_NN


env = gym.make("CartPole-v1")

n_episodes = 100000
max_t_steps = 200
discount= 0.99
ln_rate = 1e-2

actor = Actor_net(discount = discount).double()
critic = Critic_NN(discount = discount).double()

parameters = list(actor.parameters()) + list(critic.parameters())

optimiser = opt.Adam(parameters,ln_rate)

av_return = []

ac_cost=[]
cr_cost = []


for ep in range(n_episodes):

    c_state = env.reset()

    t = 0
    t_a_cost = 0
    t_c_cost = 0

    for t in range(max_t_steps):

        mean_action = actor(torch.tensor(c_state))

        d = Categorical(mean_action)

        action = d.sample()

        lp_action = d.log_prob(action)

        n_state,rwd,done,_ = env.step(action.numpy())

        advantage , critic_cost = critic.advantage(c_state,n_state,rwd,done) #

        rf_cost = actor.REINFORCE(lp_action,advantage,done)

        #critic_cost = advantage**2

        loss = rf_cost + critic_cost

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

        with torch.no_grad():
            t_a_cost += rf_cost
            t_c_cost += critic_cost

        if done:
            break

        c_state = n_state


    ac_cost.append(t_a_cost/t)
    cr_cost.append(t_c_cost/t)

    av_return.append(t)

    if ep %300 == 0:

        print("critic cost: ", sum(cr_cost) / 300)
        print("actor cost: ", sum(ac_cost)/300)
        print("av_return: ", sum(av_return)/300)
        ac_cost = []
        cr_cost = []

        av_return = []









