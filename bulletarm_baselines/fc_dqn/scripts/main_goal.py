import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm
import datetime
import threading

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
sys.path.append('./')
sys.path.append('..')
from bulletarm_baselines.fc_dqn.scripts.create_agent import createAgent
from bulletarm_baselines.fc_dqn.storage.buffer import QLearningBufferExpert, QLearningBuffer
from bulletarm_baselines.logger.logger import Logger
from bulletarm_baselines.logger.baseline_logger import BaselineLogger
from bulletarm_baselines.fc_dqn.utils.schedules import LinearSchedule
from bulletarm_baselines.fc_dqn.utils.env_wrapper import EnvWrapper

from bulletarm_baselines.fc_dqn.utils.parameters import *
from bulletarm_baselines.fc_dqn.utils.torch_utils import augmentBuffer, augmentBufferD4
from bulletarm_baselines.fc_dqn.scripts.fill_buffer_deconstruct import fillDeconstructUsingRunner

from bulletarm_baselines.fc_dqn.scripts.load_classifier import load_classifier


ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert abs_state abs_goal abs_state_next abs_goal_next')

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss

def update_abs_goals(abs_states):
    with torch.no_grad():
        zeros_goals = torch.zeros_like(abs_states)
        return torch.max(abs_states - 1, zeros_goals)

def train_step(agent, replay_buffer, logger):
    batch = replay_buffer.sample(batch_size)
    loss, td_error = agent.update(batch)
    logger.logTrainingStep(loss)
    if logger.num_training_steps % target_update_freq == 0:
        agent.updateTarget()

def saveModelAndInfo(logger, agent):
    logger.writeLog()
    logger.exportData()
    agent.saveModel(os.path.join(logger.models_dir, 'snapshot'))


def evaluate(envs, agent, logger):
  states, in_hands, obs = envs.reset()
  evaled = 0
  temp_reward = [[] for _ in range(num_eval_processes)]
  if not no_bar:
    eval_bar = tqdm(total=num_eval_episodes)
  while evaled < num_eval_episodes:
    # TODO: classifier
    abs_states = torch.tensor([1,2,3,4,5]).to(device) #self.classify_(obs, in_hands)
    abs_goals = update_abs_goals(abs_states)
    q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs,abs_states,abs_goals, 0)
    actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
    states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
    rewards = rewards.numpy()
    dones = dones.numpy()
    states = copy.copy(states_)
    in_hands = copy.copy(in_hands_)
    obs = copy.copy(obs_)
    for i, r in enumerate(rewards.reshape(-1)):
      temp_reward[i].append(r)
    evaled += int(np.sum(dones))
    for i, d in enumerate(dones.astype(bool)):
      if d:
        R = 0
        for r in reversed(temp_reward[i]):
          R = r + gamma * R
        logger.logEvalEpisode(temp_reward[i], discounted_return=R)
        # eval_rewards.append(R)
        temp_reward[i] = []
    if not no_bar:
      eval_bar.update(evaled - eval_bar.n)
  # logger.eval_rewards.append(np.mean(eval_rewards[:num_eval_episodes]))
  logger.logEvalInterval()
  logger.writeLog()
  if not no_bar:
    eval_bar.close()

def train():
    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    eval_envs = EnvWrapper(num_eval_processes, env, env_config, planner_config)
    num_objects = envs.getNumObj()
    num_classes = 2 * num_objects - 1 
    agent = createAgent(num_classes)
    eval_agent = createAgent(num_classes,test=True)

    # load classifier
    # classifier = load_classifier('1b1b1r',num_objects)


    if load_model_pre:
        agent.loadModel(load_model_pre)
    agent.train()
    eval_agent.train()

    # logging
    base_dir = os.path.join(log_pre, '{}_{}_{}'.format(alg, model, env))
    if note:
        base_dir += '_'
        base_dir += note
    if not log_sub:
        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d.%H:%M:%S')
        log_dir = os.path.join(base_dir, timestamp)
    else:
        log_dir = os.path.join(base_dir, log_sub)

    hyper_parameters['model_shape'] = agent.getModelStr()
    logger = BaselineLogger(log_dir, checkpoint_interval=save_freq, num_eval_eps=num_eval_episodes, hyperparameters=hyper_parameters, eval_freq=eval_freq)
    logger.saveParameters(hyper_parameters)

    if buffer_type == 'expert':
        replay_buffer = QLearningBufferExpert(buffer_size)
    else:
        raise NotImplementedError('buffer type in ["expert"]')
        
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)
    print(f'explore scheduler for {explore} steps, from {init_eps} to {final_eps}')

    states, in_hands, obs = envs.reset()

    if load_sub:
        logger.loadCheckPoint(os.path.join(base_dir, load_sub, 'checkpoint'), agent.loadFromState, replay_buffer.loadFromState)
    #------------------------------------- expert transition ----------------------------------------#    
    if planner_episode > 0 and not load_sub:
        if fill_buffer_deconstruct:
            fillDeconstructUsingRunner(agent, replay_buffer)
    #------------------------------------- pretrainning with expert ----------------------------------------#    

    #-------------------------------------- start trainning ----------------------------------------------#
    if not no_bar:
        pbar = tqdm(total=max_train_step)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()
    while logger.num_training_steps < max_train_step:
        if fixed_eps:
            eps = final_eps
        else:
            eps = exploration.value(logger.num_eps)
        is_expert = 0
        # TODO: classifier
        abs_states = torch.tensor([1,2,3,4,5]).to(device) #self.classify_(obs, in_hands)
        abs_goals = update_abs_goals(abs_states)

        q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(
            states, in_hands, obs,abs_states,abs_goals, eps
            )

        buffer_obs = getCurrentObs(in_hands, obs)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        envs.stepAsync(actions_star, auto_reset=False)

        if len(replay_buffer) >= training_offset:
            for training_iter in range(training_iters):
                train_step(agent, replay_buffer, logger)

        states_, in_hands_, obs_, rewards, dones = envs.stepWait()

        # TODO: classifier
        abs_states_next = torch.tensor([1,2,3,4,5]).to(device) #self.classify_(obs_, in_hands_)
        abs_goals_next =  update_abs_goals(abs_states_next)
        # goals_achieved = (abs_states_next == abs_goals)
        # rewards = goals_achieved.unsqueeze(1).float() - 1.0
        # for i in range(num_processes):
        #     if goals_achieved[i].cpu().item() is False:
        #         dones[i] = 1.0

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                in_hands_[idx] = reset_in_hands_[j]
                obs_[idx] = reset_obs_[j]

        buffer_obs_ = getCurrentObs(in_hands_, obs_)

        for i in range(num_processes):
            replay_buffer.add(
                ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], 
                                states_[i], buffer_obs_[i], 
                                dones[i], #goals_achieved[i], 
                                torch.tensor(100), torch.tensor(is_expert), 
                                abs_states[i], abs_goals[i],
                                abs_states_next[i],abs_goals_next[i])
            )
        logger.logStep(rewards.cpu().numpy(), dones.cpu().numpy())

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()
            description = 'Action Step:{}; Episode: {}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}'.format(
              logger.num_steps, logger.num_eps, logger.getAvg(logger.training_eps_rewards, 100),
              np.mean(logger.eval_eps_rewards[-2]) if len(logger.eval_eps_rewards) > 1 and len(logger.eval_eps_rewards[-2]) > 0 else 0, eps, float(logger.getCurrentLoss()),
              timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_training_steps - pbar.n)

        if logger.num_training_steps > 0 and eval_freq > 0 and logger.num_training_steps % eval_freq == 0:
            if eval_thread is not None:
                eval_thread.join()
            eval_agent.copyNetworksFrom(agent)
            eval_thread = threading.Thread(target=evaluate, args=(eval_envs, eval_agent, logger))
            eval_thread.start()

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    if eval_thread is not None:
        eval_thread.join()

    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(agent.getSaveState(), replay_buffer.getSaveState())
    envs.close()
    eval_envs.close()

if __name__ == '__main__':
    train()