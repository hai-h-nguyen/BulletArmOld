import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.fc_dqn.agents.agents_3d.base_3d import Base3D
from bulletarm_baselines.fc_dqn.utils import torch_utils


class MetaController(Base3D):
    def __init__(self,num_classes,device,lr=1e-3, gamma=0.9):
        assert num_classes != 0
        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        self.gamma =  gamma
        self.networks = []
        self.target_networks = []
        self.optimizers = []
        self.loss_calc_dict = {}

    def updateTarget(self):
        """
        hard update the target networks
        """
        for i in range(len(self.networks)):
            self.target_networks[i].load_state_dict(self.networks[i].state_dict())

    def initNetwork(self, q1):
        self.fcn = q1
        self.target_fcn = deepcopy(q1)
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.append(self.fcn)
        self.target_networks.append(self.target_fcn)
        self.optimizers.append(self.fcn_optimizer)
        self.updateTarget()

    def forwardFCN(self, in_hand, obs, target_net=False, to_cpu=False):
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        q1 = self.fcn if not target_net else self.target_fcn
        pred = q1([obs, in_hand])
        return pred

    def getEGreedyActions(self, in_hand, obs, eps, coef=0.):
        with torch.no_grad():
            pred = self.forwardFCN(in_hand, obs, to_cpu=True)
            pred = torch.argmax(pred, dim=1)
            rand = torch.tensor(np.random.uniform(0, 1, pred.size(0)))
            rand_mask = rand < eps
            for i, m in enumerate(rand_mask):
                if (m):
                    pred[i] = torch.randint(0,self.num_classes,(1,))

        return pred

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts,\
            abs_states, abs_goals, abs_states_next, abs_goals_next,env_rewards = self._loadLossCalcDict()

        with torch.no_grad():
            pred_target = self.forwardFCN(next_obs[1], next_obs[0], target_net=True)
            pred_target = pred_target.max(1)[0]
            pred_target = env_rewards + self.gamma * pred_target * non_final_masks

        pred = self.forwardFCN(obs[1], obs[0])[torch.arange(0, batch_size), abs_goals]

        q1_td_loss = F.smooth_l1_loss(pred, pred_target)
        td_loss = q1_td_loss

        with torch.no_grad():
            td_error = torch.abs(pred - pred_target)

        return td_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss,td_error = self.calcTDLoss()

        self.fcn_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item()

    def _loadBatchToDevice(self, batch):
        """
        load the input batch in list of transitions into tensors, and save them in self.loss_calc_dict. obs and in_hand
        are saved as tuple in obs
        :param batch: batch data, list of transitions
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        states = []
        images = []
        in_hands = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        next_in_hands = []
        dones = []
        step_lefts = []
        is_experts = []
        abs_states = []
        abs_goals = []
        abs_states_next = []
        abs_goals_next = []
        env_rewards = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs[0])
            in_hands.append(d.obs[1])
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs[0])
            next_in_hands.append(d.next_obs[1])
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
            abs_states.append(d.abs_state)
            abs_states_next.append(d.abs_state_next)
            abs_goals.append(d.abs_goal)
            abs_goals_next.append(d.abs_goal_next)
            env_rewards.append(d.env_reward)
        states_tensor = torch.stack(states).long().to(self.device)
        image_tensor = torch.stack(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        in_hand_tensor = torch.stack(in_hands).to(self.device)
        if len(in_hand_tensor.shape) == 3:
            in_hand_tensor = in_hand_tensor.unsqueeze(1)
        xy_tensor = torch.stack(xys).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        env_rewards_tensor = torch.stack(env_rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        next_in_hands_tensor = torch.stack(next_in_hands).to(self.device)
        if len(next_in_hands_tensor.shape) == 3:
            next_in_hands_tensor = next_in_hands_tensor.unsqueeze(1)
        dones_tensor = torch.stack(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.stack(step_lefts).to(self.device)
        is_experts_tensor = torch.stack(is_experts).bool().to(self.device)
        abs_states_tensor = torch.stack(abs_states).to(self.device)
        abs_states_next_tensor = torch.stack(abs_states_next).to(self.device)
        abs_goals_tensor = torch.stack(abs_goals).to(self.device)
        abs_goals_next_tensor = torch.stack(abs_goals_next).to(self.device)

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = (image_tensor, in_hand_tensor)
        self.loss_calc_dict['action_idx'] = xy_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = (next_obs_tensor, next_in_hands_tensor)
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor
        self.loss_calc_dict['abs_states'] = abs_states_tensor
        self.loss_calc_dict['abs_states_next'] = abs_states_next_tensor        
        self.loss_calc_dict['abs_goals'] = abs_goals_tensor
        self.loss_calc_dict['abs_goals_next'] = abs_goals_next_tensor
        self.loss_calc_dict['env_rewards'] = env_rewards_tensor

        return states_tensor, (image_tensor, in_hand_tensor), xy_tensor, rewards_tensor, next_states_tensor, \
               (next_obs_tensor, next_in_hands_tensor), non_final_masks, step_lefts_tensor, is_experts_tensor, \
                abs_states_tensor, abs_goals_tensor, abs_states_next_tensor, abs_goals_next_tensor,env_rewards_tensor

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']
        abs_states = self.loss_calc_dict['abs_states']
        abs_states_next = self.loss_calc_dict['abs_states_next']        
        abs_goals = self.loss_calc_dict['abs_goals']
        abs_goals_next = self.loss_calc_dict['abs_goals_next']
        env_rewards = self.loss_calc_dict['env_rewards']
        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks,\
             step_lefts, is_experts, abs_states, abs_goals, abs_states_next, abs_goals_next,env_rewards