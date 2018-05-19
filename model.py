
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.autograd.Variable as AG.Variable
import torch.autograd as AG

class Model(torch.nn.Module):
    def __init__(self, obs_shape, act_size, LR, momentum, cuda):
        # Number of possible actions
        self.act_size = act_size
        # Shape of the observations
        self.obs_shape = obs_shape
        # Dummy observation of the correct shape 
        # to do shape manipulations
        dummy_obs = np.ndarray(obs_shape)
        flat_obs = dummy_obs.reshape(-1)

        # Neural Network that defines the policy
        super(Model, self).__init__() 
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5) 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) 
        self.fc1 = nn.Linear(36260, 50) 
        self.fc_act = nn.Linear(50, act_size) 
        self.fc_val = nn.Linear(50, 1) 
 
        # Optimizer that performs the gradient step 
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=LR, momentum=momentum) 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR) 
 
        # If cuda is enabled, all vars are put on the gpu 
        self.cuda_bool = cuda 
        if(self.cuda_bool): 
            self.cuda() 
 
 
    def forward(self, x): 
        """Takes in an observation and returns action probabilities and 
        an estimate of the maximum discounted reward attainable  
        from the current state""" 
 
        x = F.relu(F.max_pool2d(self.conv1(x.permute(0,3,1,2)), 2)) 
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = x.view(-1, 36260) 
        x = F.relu(self.fc1(x)) 
        act = self.fc_act(x) 
        act = F.softmax(act) 
        val = self.fc_val(x) 
        return act, val  
 
 
    def act_probs(self, obs): 
        """Returns the action probabilities given an observation""" 
 
        obs_torch = AG.Variable(obs.unsqueeze(0)) 
        policy_output, reward_estimate = self(obs_torch.float()) 
        return policy_output.data 
 
    def act_stochastic(self, obs): 
        """Returns an action chosen semi stochastically""" 
 
        act_probs = self.act_probs(obs) 
        distrib = torch.distributions.Categorical(probs=act_probs)
# samples from the categorical distribution to determine action to take 
        act_taken = distrib.sample() 
        act_taken_v = torch.zeros(self.act_size) 
        act_taken_v[act_taken[0]] = 1 
        return act_taken, act_taken_v, act_probs 
 
    def learn(self, replay_buffer): 
        """Performs backprop w.r.t. the replay buffer""" 
         
        # Calculates the discounted reward 
        #discounted_reward = replay_buffer.discount(0.99) 
        # Performs a foward step through the model 
        act_probs, expected_rewards = self(AG.Variable(replay_buffer.observations)) 
        # Advantage (diff b/t the actual discounted reward and the expected) 
        advantage = AG.Variable(replay_buffer.discount(0.99)) - expected_rewards 
        advantage_no_grad = advantage.detach() 
        # Cross Entropy where p is true distribution and q is the predicted 
        # cross_entropy = -(p*torch.log(q)).sum() 
        cross_entropy = -(AG.Variable(replay_buffer.actions)*torch.log(act_probs)).sum(dim=1) 
        # Calculate entropy of policy output 
        policy_entropy = -(act_probs*torch.log(act_probs)).sum(dim=1).mean() 
        # Policy loss (encourages behavior in buffer if advantage is positive and vice-a-versa 
        policy_loss = (advantage_no_grad*cross_entropy).mean() 
        # Critic loss (same as advantage)  
        critic_loss = (advantage**2).mean() 
        # Sums the individual losses 
        #total_loss = policy_loss + 0.25*critic_loss 
        #total_loss = critic_loss 
        #total_loss = policy_loss + 0.25*critic_loss - policy_entropy 
        total_loss = policy_loss + 0.25*critic_loss - 0.1*policy_entropy 
 
        # Debugging 
        print("Policy:", policy_loss.data.cpu().numpy()[0], "\tCritic: ", critic_loss.data.cpu().numpy()[0], "\tTotalLoss: ", total_loss.data.cpu().numpy()[0]) 
 
 
        # Clears Gradients 
        self.optimizer.zero_grad() 
        # Calculates gradients w.r.t. all weights in the model 
        total_loss.backward() 
        # Clips gradients 
        torch.nn.utils.clip_grad_norm(self.parameters(), 40) 
        # Applies the gradients 
        self.optimizer.step() 
 
        # Returns values for summary writer 
        return policy_loss.data.cpu().numpy()[0], critic_loss.data.cpu().numpy()[0], total_loss.data.cpu().numpy()[0], replay_buffer.rewards.mean(), cross_entropy.data, advantage.data 
