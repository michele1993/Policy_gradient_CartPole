import torch
import torch.nn.functional as F
import torch.nn as nn




class Actor_net(nn.Module):

    def __init__(self,discount, Input_size=4, Hidden_size=20, output_size=2):

        super().__init__()

        self.l1 = nn.Linear(Input_size, Hidden_size)
        self.l2 = nn.Linear(Hidden_size,output_size)

        #self.I = 1

        self.discount = discount

    def forward(self, x):

        x = F.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x)) #F.softmax

        return x #.view(-1,1)

    def REINFORCE(self, p_action, advantage): # , done


        Policy_cost = - p_action * advantage.detach() #* self.I # need minus because need to perform gradient ascent

        # if done:
        #     self.I = 1
        # else:
        #     self.I *= self.discount

        #print(Policy_cost.grad_fn)


        return Policy_cost