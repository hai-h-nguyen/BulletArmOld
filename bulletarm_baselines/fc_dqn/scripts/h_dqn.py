### train ###
import time
import os
import copy
import argparse
import numpy as np
import torch
from tqdm import tqdm as tqdm

from envs.multigoalenv import createMultiGoalEnv
from envs.goal_space import StructuredGoalSpace
from src.symbolic_encoder import SymbolicEncoder
from src.q_function import QFunction
from src.train import Trainer, EpisodeData
from src.h_dqn.meta_q_function import MetaQFunction
from src.memory import create_memory, AbstractMemory
from src.parallelization import RLRunner


class HDQNTrainer(Trainer):
    def __init__(self,
                 fdir,
                 env_configs,
                 device,
                 sym_encoder_lr,
                 buffer_size,
                 goal_label_method,
                 q_opt_cycles,
                 enc_opt_cycles,
                 sym_encoder_img_size,
                 batch_size,
                 gamma,
                 reward_style,
                 n_envs,
                 env_reset_goal_distribution,
                 time_to_reach_subgoal,
                 perlin_noise=0,
                 add_smoothing=0,
                 specific_goal=None,
                 use_local_policy=True,
                 detached_mode=True,
                 q_target_update_freq=1000,
                 meta_q_target_update_freq=1000,
                 **kwargs,
                ):
        super().__init__(fdir, env_configs, device, sym_encoder_lr, buffer_size,
                         goal_label_method, q_opt_cycles, enc_opt_cycles,
                         sym_encoder_img_size, batch_size,  gamma,
                         reward_style, n_envs, env_reset_goal_distribution,
                         time_to_reach_subgoal, perlin_noise, add_smoothing,
                         specific_goal, use_local_policy, detached_mode,
                          **kwargs)

        _, s_enc, _ = self.envs.reset()
        self.meta_memory = AbstractMemory(self.ep_length, s_enc, buffer_size)

        self.meta_q_function = self.create_meta_q_function(gamma, reward_style,
                                                           meta_q_target_update_freq)

    def create_meta_q_function(self, gamma, reward_style, target_update_freq):
        obs, s_enc, _ = self.envs.reset()
        enc_shape = self._enc_to_tensor(s_enc)[0].shape
        # load default kw_args
        return MetaQFunction(enc_shape,
                             gamma=gamma,
                             reward_style=reward_style,
                             target_update_freq=target_update_freq,
                            ).to(self.device)

    def set_new_subgoals(self, ep_data, subg_r, s_enc, subg_enc, g_enc, r, add_to_memory=True):
        mask = ep_data.n_actions_on_sg == self.time_to_reach_subgoal
        ep_data.n_actions_on_sg[mask == 1] = 0
        mask = np.bitwise_or(r.flatten() == 1, mask)
        if mask.any():
            env_nums = np.where(mask)[0]
            _sg_enc = self.get_metagoal((None, s_enc[env_nums], None, g_enc[env_nums]),
                                        self.meta_epsilon)
            if add_to_memory:
                self.meta_memory.add_to_episodes(env_nums, s_enc[env_nums],
                                                 _sg_enc, r[env_nums])
            subg_enc[env_nums] = _sg_enc.copy()

        return ep_data, subg_enc

    def reset_as_needed(self,
                        ep_data,
                        transition,
                        env_reset_goal_distribution=None,
                        goal_subset=None,
                       ):
        if env_reset_goal_distribution is None:
            env_reset_goal_distribution = self.env_reset_goal_distribution
        if goal_subset is None and self.specific_goal is not None:
            goal_subset = [self.specific_goal]

        mask = ep_data.active_envs==0
        obs, s_enc, subg_enc, g_enc = transition
        if mask.any():
            # reset environments that need to be reset
            env_nums_to_reset = np.where(mask)[0]
            _obs, _s_enc, _g_enc = self.envs.reset_envs(env_nums_to_reset,
                                                        env_reset_goal_distribution,
                                                        goal_subset)
            _path_length = self.envs.opt_path_length(env_nums_to_reset)
            _subg_enc = self.get_metagoal((None, _s_enc, None, _g_enc),
                                          self.meta_epsilon)
            self.memory.start_episodes(env_nums_to_reset, _obs,
                                       _s_enc, _g_enc)
            self.meta_memory.start_episodes(env_nums_to_reset, _s_enc, _g_enc)

            for i in range(len(obs)):
                obs[i][env_nums_to_reset] = _obs[i].copy()
            s_enc[env_nums_to_reset] = _s_enc.copy()
            g_enc[env_nums_to_reset] = _g_enc.copy()
            subg_enc[env_nums_to_reset] = _subg_enc.copy()

            ep_data.n_subgoals_reached[env_nums_to_reset] = 0
            ep_data.n_subgoals_total[env_nums_to_reset] = _path_length.copy()
            ep_data.n_actions_on_sg[env_nums_to_reset] = 0
            ep_data.n_actions_taken[env_nums_to_reset] = 0
            ep_data.active_envs[:] = True

        return ep_data, (obs, s_enc, subg_enc, g_enc)

    def run_and_yield(self,
                      env_step_interval,
                      total_env_steps,
                      eps_range=(1.0, 0.0),
                      goal_subset=None,
                      novelty_eps=False,
                      optimal_period=0,
                      log_freq=1000):
        pretrain_period = int(total_env_steps * 0.2)
        step_count = 0
        episode_count = 0
        log_mod_counter = 0
        step_mod_counter = 0
        self.total_env_steps = total_env_steps

        rewards_data = []

        #initialize data
        ep_data = EpisodeData(self.n_envs)
        obs, s_enc, g_enc = self.envs.reset(goal_distribution=self.env_reset_goal_distribution,
                                            goal_subset=goal_subset)
        subg_enc = np.copy(g_enc)

        meta_anneal_period = total_env_steps - optimal_period - pretrain_period

        while step_count <= total_env_steps:
            self.meta_epsilon = np.clip(1 - (step_count - pretrain_period)/meta_anneal_period, 0, 1)

            full_state = (obs, s_enc, subg_enc, g_enc)
            ep_data, full_state = self.reset_as_needed(ep_data, full_state,
                                                      goal_subset=goal_subset)

            step_progress = min(step_count/(total_env_steps-optimal_period), 1)
            epsilon = eps_range[0]*(1-step_progress) + eps_range[1]*step_progress

            a = self.get_action(full_state, epsilon)

            # for speed, we step env async and optimize while it runs
            self.envs.stepAsync(a)
            [self.optimize_models('q_function') for _ in range(self.q_opt_cycles)]

            obs, s_enc, subg_enc, g_enc = full_state
            obs, s_enc, done = self.envs.stepWait()
            ep_data.n_actions_on_sg += 1
            ep_data.n_actions_taken += 1
            step_count += self.n_envs
            log_mod_counter += self.n_envs
            step_mod_counter += self.n_envs

            # calculate rewards
            subg_r = self.compute_reward(s_enc, subg_enc, done, obs[0])
            r = self.compute_reward(s_enc, g_enc, done, obs[0])

            self.memory.add_to_episodes(np.arange(self.n_envs), obs, s_enc,
                                        subg_enc, a, subg_r, done,
                                        ep_data.n_subgoals_reached[:,None]+subg_r)


            # determine what envs should be reset
            ep_data = self.check_for_envs_to_reset(ep_data, r, done)

            terminal_env_nums = np.where(ep_data.active_envs == 0)[0]
            self.memory.end_episodes(terminal_env_nums)
            self.meta_memory.end_episodes(terminal_env_nums, s_enc[terminal_env_nums],
                                          subg_enc[terminal_env_nums],
                                          r[terminal_env_nums])
            episode_count += len(terminal_env_nums)

            # change subgoals as needed
            ep_data, subg_enc = self.set_new_subgoals(ep_data, subg_r, s_enc,
                                                      subg_enc, g_enc, r)

            rewards_data.extend([(step_count,
                                  ep_data.n_actions_taken[i],
                                  r[i,0]*ep_data.n_subgoals_total[i],
                                  ep_data.n_subgoals_total[i],
                                 ) for i in terminal_env_nums])

            if log_mod_counter > log_freq:
                log_mod_counter = log_mod_counter % log_freq
                self.q_function.log_progress(self.fdir)
                self.sym_encoder.log_progress(self.fdir)
                np.save(f"{self.fdir}/rewards.npy", np.array(rewards_data).T)

            if step_mod_counter >= env_step_interval:
                step_mod_counter = step_mod_counter % env_step_interval
                yield step_count

            [self.optimize_models('encoder') for _ in range(self.enc_opt_cycles)]

    def optimize_models(self, models='both'):
        assert models in ('q_function', 'encoder', 'both')
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
        else:
            (s,a,sp,r,meta_r,d) = batch
        if models in ('q_function', 'both'):
            holding, ih, td, sv, s_enc, sg_enc, g_enc = s
            s = (holding, ih, td, sv, s_enc, sg_enc)
            next_holding, next_ih, next_td, next_sv, next_s_enc, next_sg_enc, next_g_enc = sp
            sp = (next_holding, next_ih, next_td, next_sv, next_s_enc, next_sg_enc)
            self.q_function.optimize((self._state_to_tensor(s),
                                      self._action_to_tensor(a),
                                      self._reward_to_tensor(r),
                                      self._state_to_tensor(sp),
                                      self._reward_to_tensor(d),
                                     ))
            meta_batch = self.meta_memory.sample(self.batch_size)
            if meta_batch is not None:
                s, sg, sp, g, r = meta_batch
                self.meta_q_function.optimize((self._enc_to_tensor(s),
                                               sg,
                                               self._enc_to_tensor(sp),
                                               self._enc_to_tensor(g),
                                               self._reward_to_tensor(r),
                                              ))
        if models in ('encoder', 'both'):
            self.sym_encoder.optimize(self._img_to_tensor(s[3]),
                                      self._enc_to_tensor(s[4]))
            self.sym_encoder.optimize(self._img_to_tensor(sp[3]),
                                      self._enc_to_tensor(sp[4]))

    def eval_q_function(self, reset_goal_distribution='all',
                        goal_subset=None, n_runs=100,
                        time_to_reach_subgoal=4):
        eval_data = {'terminal' : [0,0],
                     'all' : [0,0]}
        progress_data = []

        # initialize data
        ep_data = EpisodeData(self.n_envs)
        obs, s_enc, g_enc = self.envs.reset(reset_goal_distribution,
                                            goal_subset)
        subg_enc = np.copy(g_enc)

        total_rewards = 0
        t = time.time()
        self.meta_epsilon = 0
        while len(progress_data) < n_runs:
            full_state = (obs, s_enc, subg_enc, g_enc)
            ep_data, full_state = self.reset_as_needed(ep_data, full_state,
                                                       reset_goal_distribution,
                                                       goal_subset)

            # pick action with no exploration noise added
            a = self.get_action(full_state, epsilon=0.0)

            obs, s_enc, done = self.envs.step(a)
            ep_data.n_actions_on_sg += 1
            ep_data.n_actions_taken += 1

            # calculate rewards
            subg_r = self.compute_reward(s_enc, subg_enc, done, obs[0])
            
            r = self.compute_reward(s_enc, g_enc, done, obs[0])


            # change subgoals as needed
            ep_data, subg_enc = self.set_new_subgoals(ep_data, subg_r, s_enc,
                                                      subg_enc, g_enc, r, False)

            # Check to see what envs must be reset
            ep_data = self.check_for_envs_to_reset(ep_data, r, done,
                                                   time_to_reach_subgoal)

            # push stats to performance array
            for env_num in np.where(ep_data.active_envs==0)[0]:
                progress_data.append((ep_data.n_subgoals_reached[env_num],
                                      ep_data.n_subgoals_total[env_num]))

                eval_data['all'][0] += r[env_num][0]
                eval_data['all'][1] += 1
                if self.envs.is_terminal_goal(env_num):
                    eval_data['terminal'][0] += r[env_num][0]
                    eval_data['terminal'][1] += 1

        return eval_data, np.array(progress_data)

    def get_metagoal(self, full_state, epsilon=0.0):
        _, s_enc, _, g_enc = full_state

        t_s_enc = self._enc_to_tensor(s_enc)
        t_g_enc = self._enc_to_tensor(g_enc)

        return self.meta_q_function.goal_selection(t_s_enc, t_g_enc, epsilon)

    def check_for_envs_to_reset(self, ep_data, r, done,
                                time_to_reach_subgoal=None,
                               ):
        if time_to_reach_subgoal is None:
            time_to_reach_subgoal = self.time_to_reach_subgoal
            ep_length = self.ep_length
        else:
            ep_length = ep_data.n_subgoals_total * time_to_reach_subgoal

        out_of_steps = ep_data.n_actions_taken >= ep_length

        terminal_envs = (r.flatten()+out_of_steps+done.flatten()) > 0
        ep_data.active_envs[terminal_envs] = 0

        return ep_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detached', action="store_true")
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    args = parser.parse_args()

    training_configs = load_training_configs(args.config_file)
    env_configs = load_default_env_configs()
    if 'env' in training_configs:
        other_env_configs = training_configs.pop('env')
        # overwrite any configs
        env_configs = {**env_configs, **other_env_configs}

    fdir = os.path.dirname(args.config_file)
    trial_name = fdir.split('/')[-1]

    trainer = Trainer(fdir=fdir,
                      device=args.device,
                      detached_mode=args.detached,
                      env_configs=env_configs,
                      **training_configs,
                     )
    trainer._log(f"=== STARTING trial:'{trial_name}' on device:'{args.device}'" \
          f" @{time.strftime('%b %d %H:%M:%S', time.localtime())} ===")
    if args.detached:
        trainer._log(' --- running in detached mode --- ')

    n_episodes = training_configs.get('n_episodes',50000)
    log_freq = training_configs.get('log_freq', 500)
    eval_freq = training_configs.get('eval_freq', 5000)
    trainer._log(f"running q-learning for {n_episodes}...")
    trainer.train(n_episodes, log_freq, eval_freq)

    trainer._log(f"=== FINISHED trial:'{trial_name}' on device:'{args.device}'" \
          f" @{time.strftime('%b %d %H:%M:%S', time.localtime())} ===")


### meta dqn ###

import numpy as np
import numpy.random as npr
import torch.nn as nn
import time

class MetaNet(nn.Module):
    def __init__(self, enc_shape):
        super().__init__()

        
        _goals = np.product(enc_shape) - 1
        self.layers = nn.Sequential(
 [None[
     nn.Linear(2*np.product(enc_shape), 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, n_goals),
        )
        self.criterion = nn.MSELoss()

    def forward(self, s_enc, g_enc):
                        target_update_freq=1000,
                 ):
        #unpack arguments
        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        assert reward_style in ('positive', 'negative')
        self.reward_style = reward_style

        self.n_goals = np.product(enc_shape) - 1

        #set up networks
        self.network = MetaNet(enc_shape)
        self.target = MetaNet(enc_shape)

        self._hard_target_update()
        self.target.eval()
        self.optim = torch.optim.Adam(self.network.parameters(),
                                      lr=self.lr)

        self.opt_steps = 0
        self.loss_history = []
        self.reward_history = []

    def log_progress(self, fdir, addendum=''):
        # save avg rewards
        rewards_data = np.vstack((np.arange(len(self.reward_history)),
                                 self.reward_history))
        np.save(f"{fdir}/meta_q_function_rewards.npy", rewards_data)

        loss_data = np.vstack((np.arange(len(self.loss_history)),
                                 self.loss_history))
        np.save(f"{fdir}/meta_q_function_loss.npy", loss_data)
        if addendum != '':
            torch.save(self.network.state_dict(), f"{fdir}/meta_q_function{addendum}.pt")

    def goal_selection(self, s_enc, g_enc, epsilon=0.0):
        goals = np.zeros((len(s_enc),2), dtype=int)
        if npr.random() < epsilon:
            # add 1 because 0 means empty platform

    def add_robot():
        height = 0.2
        np.save(f"{fdir}/meta_q_function_loss.npy", loss_data)
        if addendum !=08'
            torch.save(lf.networkheight, f"{fdir}/meta_q_function{addendum}.pt")

    def goal_selection((0,0,hei), s_enc, g_enc, epsilon=0.0):
            goals[:,0] = npr.randint(self.n_goals, size=len(s_enc))
        else:
            with torch.no_grad():
                out = self.network.forward(s_enc, g_enc)
                goals[:,0] = torch.max(out, 1)[1].cpu().numpy()

                add_robot()

        return goals

    def optimize(self, batch=None):
        if batch is None:
            return

        self.opt_steps += 1

        S_ENC, SG_id, SP_ENC, G_ENC, R = batch
        R = R.squeeze()
        batch_size = len(R)

        Qvals_pred = self.network(S_ENC, G_ENC)
        Q_pred = Qvals_pred[np.arange(batch_size),SG_id[:,0].astype(int)]
        with torch.no_grad():
            Qvals_next = self.target(SP_ENC, G_ENC)
            # double Q update
            G_next = self.network(SP_ENC, G_ENC).max(1)[1].unsqueeze(1)
            Q_next = Qvals_next.gather(1,G_next).squeeze()
            if self.reward_style == 'positive':
                # Q_target = R + self.gamma * Q_next*(1-Done)
                Q_target = R + self.gamma * Q_next * (1-R)
            elif self.reward_style == 'negative':
                Q_target = (R-1) + self.gamma * Q_next * (1-R)
            elif self.reward_style == 'stepwise':
                Q_target = R + self.gamma * Q_next

        self.optim.zero_grad()
        loss = self.network.loss(Q_pred, Q_target)
        loss.backward()
        for param in self.network.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1,1)
        self.optim.step()

        if self.opt_steps > self.target_update_freq:
            self._hard_target_update()
            self.opt_steps = 0

        self.loss_history.append(loss.mean().item())

    def load_network(self, fdir):
        self.network.load_state_dict(torch.load(fdir+"/meta_q_function.pt"))
        self._hard_target_update()

    def _hard_target_update(self):
        self.target.load_state_dict(self.network.state_dict())

    def to(self, device):
        self.network.to(device)
        self.target.to(device)
        return self

    def eval(self):
        self.network.eval()
