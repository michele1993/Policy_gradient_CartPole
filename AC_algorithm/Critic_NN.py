import torch.nn as nn
import torch
class Critic_NN(nn.Module):

    def __init__(self,discount, n_inputs = 4, n_outputs=1):

        super().__init__()
        self.discount= discount


        self.l1 = nn.Linear(n_inputs,n_outputs)

        self.I = 1

    def forward(self, x):

        x = self.l1(x)

        return x

    def advantage(self,t1,t2,rwd, done):

        vs_1 = self(torch.tensor(t1))

        if done:
            vs_2 = 0

        else:
            vs_2 = self(torch.tensor(t2)).detach()


        td_error = (rwd + self.discount * vs_2  - vs_1).detach()

        loss =  self.I * td_error * vs_1

        if done:
            self.I = 1
        else:
            self.I *= self.discount

        return vs_1, loss



