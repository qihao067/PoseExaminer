import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.distributions
import numpy as np

# 2-step - ori
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_dim, num_imgs, std_limit=3.0, ini_gap=0.01, withOcc=False):
        super(GaussianPolicy, self).__init__()

        self.num_dim = num_dim
        self.num_imgs_phase1 = 1
        self.num_imgs_phase2 = num_imgs
        self.num_outputs = self.num_dim # * self.num_imgs
        self.pose_dim = 63
        self.withOcc = withOcc

        # # Zgen
        # self.noise_rand_top = Variable(torch.ones(self.num_imgs_phase1, self.num_dim)*ini_gap,requires_grad=True) # should > 0
        # self.noise_rand_bottom = Variable(torch.ones(self.num_imgs_phase1, self.num_dim)*ini_gap,requires_grad=True) # should > 0
        # Poses
        self.noise_rand_top = Variable(torch.ones(self.num_imgs_phase1, self.pose_dim)*ini_gap,requires_grad=True) # should > 0
        self.noise_rand_bottom = Variable(torch.ones(self.num_imgs_phase1, self.pose_dim)*ini_gap,requires_grad=True) # should > 0

        if self.withOcc:
            ini_position = torch.tensor((1080.0/2,1080.0/2,0.0))
            self.occParm = Variable(torch.ones(3)*ini_position,requires_grad=True)

        self.hid_size = [32]

        self.fc_action = nn.Linear(num_inputs, self.num_outputs)

        self.action_log_std = nn.Parameter(torch.ones(self.num_imgs_phase1, self.num_dim)) * 0.05
        self.std_limit = std_limit

        self.train()
        self.init_weights()
    
    def init_weights(self):
        bound = 1
        nn.init.constant_(self.fc_action.weight.data, 0.0)
        nn.init.constant_(self.fc_action.bias.data, 0.0)
        

    def forward(self, inputs, phase=1):
        self.fc_action.weight.data = torch.clamp(self.fc_action.weight.data, -self.std_limit, self.std_limit)
        mean = self.fc_action(inputs)

        std = self.action_log_std.expand_as(mean).cuda()
        mean = torch.clamp(mean, -self.std_limit, self.std_limit)
    

        low = self.noise_rand_bottom.cuda()
        high = self.noise_rand_top.cuda()
        return mean, std, low, high

    def act(self, inputs, phase=1):
        mean, std, low, high = self(inputs, phase)

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        gap = high+low

        action_log_probs = dist.log_prob(action)
        action_log_probs = torch.sum(action_log_probs, dim=1 , keepdim=True)

        self.action_log_std.detach_()

        if self.withOcc:
            occ_position = self.occParm.cuda()
            return action, action_log_probs, gap, mean, std, low, high, occ_position
        else:
            return action, action_log_probs, gap, mean, std, low, high

