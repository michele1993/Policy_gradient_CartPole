import torch.nn as nn
import torch
import torch.nn.functional as F

class Critic_NN(nn.Module):

    def __init__(self,discount, n_inputs = 4,n_hiddens = 20, n_outputs=1):

        super().__init__()
        self.discount= discount

        self.l1 = nn.Linear(n_inputs,n_hiddens)
        self.l2 = nn.Linear(n_hiddens, n_outputs)

        self.I = 1

    def forward(self, x):

        x = F.relu(self.l1(x))
        x= self.l2(x)

        return x

    def advantage(self,t1,t2,rwd, done):

        vs_1 = self(torch.tensor(t1))

        if done:
            vs_2 = 0

        else:
            vs_2 = self(torch.tensor(t2)).detach()


        td_error = (rwd + self.discount * vs_2  - vs_1)

        #loss =   td_error * vs_1 # * self.I

        # if done:
        #     self.I = 1
        # else:
        #     self.I *= self.discount

        return td_error



