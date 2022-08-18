from math import ceil
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.ticker import FuncFormatter

def getRewardsSingle(rewards, window=1000):
    moving_avg = []
    i = window
    while i-window < len(rewards):
        moving_avg.append(np.average(rewards[i-window:i]))
        i += window

    moving_avg = np.array(moving_avg)
    return moving_avg

def plotLearningCurveAvg(rewards, window=1000, label='reward', color='b', shadow=True, ax=plt, legend=True, linestyle='-'):
    min_len = np.min(list(map(lambda x: len(x), rewards)))
    rewards = list(map(lambda x: x[:min_len], rewards))
    avg_rewards = np.mean(rewards, axis=0)
    # avg_rewards = np.concatenate(([0], avg_rewards))
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    # std_rewards = np.concatenate(([0], std_rewards))
    xs = np.arange(window, window * (avg_rewards.shape[0]+1), window)
    if shadow:
        ax.fill_between(xs, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    if legend:
        ax.legend(loc=4)
    return l

def plotEvalCurveAvg(rewards, freq=1000, label='reward', color='b', shadow=True, ax=plt, legend=True, linestyle='-'):
    min_len = np.min(list(map(lambda x: len(x), rewards)))
    rewards = list(map(lambda x: x[:min_len], rewards))
    avg_rewards = np.mean(rewards, axis=0)
    # avg_rewards = np.concatenate(([0], avg_rewards))
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    # std_rewards = np.concatenate(([0], std_rewards))
    xs = np.arange(freq, freq * (avg_rewards.shape[0]+1), freq)
    if shadow:
        ax.fill_between(xs, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    if legend:
        ax.legend(loc=4)
    return l

def plotEvalCurve(base, step=50000, use_default_cm=False, freq=1000):
    # plt.style.use('ggplot')
    plt.style.use('seaborn-whitegrid')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            # for open loop
            'Equi_ASR_SDQfD': 'b',
            'CNN_ASR_SDQfD': 'r',
            'CNN_RotFCN_SDQfD': 'g',
            'CNN_FCN_SDQfD': 'purple',
            # for close loop
            'Equi_SACfD_PER_Aug': 'b',
            'Equi_SACfD': 'r',
            'CNN_SACfD': 'g',
            'FERM': 'purple',
            # 'Equi_SAC': 'orange',
            
        }

    linestyle_map = {
    }
    name_map = {
        # for open loop
        'Equi_ASR_SDQfD': 'Equi ASR',
        'CNN_ASR_SDQfD': 'CNN ASR',
        'CNN_RotFCN_SDQfD': 'Rot FCN',
        'CNN_FCN_SDQfD': 'FCN',
        # for close loop
        'CNN_SACfD': 'CNN SACfD',
        'Equi_SAC': 'Equi SAC',
        'Equi_SACfD': 'Equi SACfD',
        'FERM': 'FERM',
        'Equi_SACfD_PER_Aug': 'Equi SACfD+PER+Aug',

    }

    sequence = {
        # for open loop
        'Equi_ASR_SDQfD': '1',
        'CNN_ASR_SDQfD': '2',
        'CNN_RotFCN_SDQfD': '3',
        'CNN_FCN_SDQfD': '4',
        # for close loop
        'CNN_SACfD': '3',
        # 'Equi_SAC': '3',
        'Equi_SACfD': '2',
        'FERM': '5',
        'Equi_SACfD_PER_Aug': '1',
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                # r = np.load(os.path.join(base, method, run, 'info/eval_rewards.npy'))
                data = pickle.load(open(os.path.join(base, method, run, 'log_data.pkl'), 'rb'))
                rewards = data['eval_eps_rewards']
                r = [np.mean(x) for x in rewards[:-1]]
                rs.append(r[:step//freq])
            except Exception as e:
                print(e)
                continue

        plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of training steps')
    # if base.find('bbp') > -1:
    plt.ylabel('eval success rate')
    # plt.xlim((-100, step+100))
    plt.yticks(np.arange(0., 1.05, 0.1))
    # plt.ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(base, f'{base}.png'), bbox_inches='tight',pad_inches = 0)

def plotStepRewardCurve(base, step=50000, use_default_cm=False, freq=1000, file_name='step_reward'):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'dpos=0.05, drot=0.25pi': 'b',
            'dpos=0.05, drot=0.125pi': 'g',
            'dpos=0.03, drot=0.125pi': 'r',
            'dpos=0.1, drot=0.25pi': 'purple',

            'ban0': 'g',
            'ban2': 'r',
            'ban4': 'b',
            'ban8': 'purple',
            'ban16': 'orange',

            'C4': 'g',
            'C8': 'r',
            'D4': 'b',
            'D8': 'purple',

            '0': 'r',
            '10': 'g',
            '20': 'b',
            '40': 'purple',

            'sac+ban4': 'b',
            'sac+rot rad': 'g',
            'sac+rot rad+ban4': 'r',
            'sac+ban0': 'purple',

            'sac+aux+ban0': 'g',
            'sac+aux+ban4': 'r',

            'equi sac': 'b',
            'ferm': 'g'
        }

    linestyle_map = {
    }
    name_map = {
        'ban0': 'buffer aug 0',
        'ban2': 'buffer aug 2',
        'ban4': 'buffer aug 4',
        'ban8': 'buffer aug 8',
        'ban16': 'buffer aug 16',

        'sac+ban4': 'SAC + buffer aug',
        'sac+rot rad': 'SAC + rot RAD',
        'sac+rot rad+ban4': 'SAC + rot RAD + buffer aug',
        'sac+ban0': 'SAC',

        'sac+aux+ban4': 'SAC + aux loss + buffer aug',
        'sac+aux+ban0': 'SAC + aux loss',

        'sac': 'SAC',
        'sacfd': 'SACfD',

        'sac+crop rad': 'SAC + crop RAD',

        'equi sac': 'Equivariant SAC',
        'ferm': 'FERM'
    }

    sequence = {
        'ban0': '0',
        'ban2': '1',
        'ban4': '2',
        'ban8': '3',
        'ban16': '4',

        'sac+ban0': '0',
        'sac+ban4': '1',
        'sac+aux+ban0': '2',
        'sac+aux+ban4': '3',
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                step_reward = np.load(os.path.join(base, method, run, 'info/{}.npy'.format(file_name)))
                r = []
                for k in range(1, step+1, freq):
                    window_rewards = step_reward[(k <= step_reward[:, 0]) * (step_reward[:, 0] < k + freq)][:, 1]
                    if window_rewards.shape[0] > 0:
                        r.append(window_rewards.mean())
                    else:
                        break
                    # r.append(step_reward[(i <= step_reward[:, 0]) * (step_reward[:, 0] < i + freq)][:, 1].mean())
                rs.append(r)
            except Exception as e:
                print(e)
                continue

        plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of training steps')
    # if base.find('bbp') > -1:
    plt.ylabel('discounted reward')
    # plt.xlim((-100, step+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))
    # plt.ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'step_reward.png'), bbox_inches='tight',pad_inches = 0)

def plotStepSRCurve(base, step=50000, use_default_cm=False, freq=1000, file_name='step_success_rate'):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
        }

    linestyle_map = {
    }
    name_map = {
    }

    sequence = {
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                step_reward = np.load(os.path.join(base, method, run, 'info/{}.npy'.format(file_name)))
                r = []
                for k in range(1, step+1, freq):
                    window_rewards = step_reward[(k <= step_reward[:, 0]) * (step_reward[:, 0] < k + freq)][:, 1]
                    if window_rewards.shape[0] > 0:
                        r.append(window_rewards.mean())
                    else:
                        break
                    # r.append(step_reward[(i <= step_reward[:, 0]) * (step_reward[:, 0] < i + freq)][:, 1].mean())
                rs.append(r)
            except Exception as e:
                print(e)
                continue

        plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of training steps')
    # if base.find('bbp') > -1:
    plt.ylabel('success rate')
    # plt.xlim((-100, step+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))
    # plt.ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'step_sr.png'), bbox_inches='tight',pad_inches = 0)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def plotLearningCurve(base, ep=50000, use_default_cm=False, window=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'equi+bufferaug': 'b',
            'cnn+bufferaug': 'g',
            'cnn+rad': 'r',
            'cnn+drq': 'purple',
            'cnn+curl': 'orange',
        }

    linestyle_map = {

    }
    name_map = {
        'equi+bufferaug': 'Equivariant',
        'cnn+bufferaug': 'CNN',
        'cnn+rad': 'RAD',
        'cnn+drq': 'DrQ',
        'cnn+curl': 'FERM',
    }

    sequence = {
        'equi+equi': '0',
        'cnn+cnn': '1',
        'cnn+cnn+aug': '2',
        'equi_fcn_asr': '3',
        'tp': '4',

        'equi_fcn': '0',
        'fcn_si': '1',
        'fcn_si_aug': '2',
        'fcn': '3',

        'equi+deictic': '2',
        'cnn+deictic': '3',

        'q1_equi+q2_equi': '0',
        'q1_equi+q2_cnn': '1',
        'q1_cnn+q2_equi': '2',
        'q1_cnn+q2_cnn': '3',

        'q1_equi+q2_deictic': '0.5',
        'q1_cnn+q2_deictic': '4',

        'equi_fcn_': '1',

        '5l_equi_equi': '0',
        '5l_equi_deictic': '1',
        '5l_equi_cnn': '2',
        '5l_cnn_equi': '3',
        '5l_cnn_deictic': '4',
        '5l_cnn_cnn': '5',

    }

    # house1-4
    # plt.plot([0, 100000], [0.974, 0.974], label='expert', color='pink')
    # plt.axvline(x=10000, color='black', linestyle='--')

    # house1-5
    # plt.plot([0, 50000], [0.974, 0.974], label='expert', color='pink')
    # 0.004 pos noise
    # plt.plot([0, 50000], [0.859, 0.859], label='expert', color='pink')

    # house1-6 0.941

    # house2
    # plt.plot([0, 50000], [0.979, 0.979], label='expert', color='pink')
    # plt.axvline(x=20000, color='black', linestyle='--')

    # house3
    # plt.plot([0, 50000], [0.983, 0.983], label='expert', color='pink')
    # plt.plot([0, 50000], [0.911, 0.911], label='expert', color='pink')
    # 0.996
    # 0.911 - 0.01

    # house4
    # plt.plot([0, 50000], [0.948, 0.948], label='expert', color='pink')
    # plt.plot([0, 50000], [0.862, 0.862], label='expert', color='pink')
    # 0.875 - 0.006
    # 0.862 - 0.007 *
    # stack
    # plt.plot([0, 100000], [0.989, 0.989], label='expert', color='pink')
    # plt.axvline(x=10000, color='black', linestyle='--')

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                if method.find('BC') >= 0 or method.find('tp') >= 0:
                    rs.append(r[-window:].mean())
                else:
                    rs.append(getRewardsSingle(r[:ep], window=window))
            except Exception as e:
                print(e)
                continue

        if method.find('BC') >= 0 or method.find('tp') >= 0:
            avg_rewards = np.mean(rs, axis=0)
            std_rewards = stats.sem(rs, axis=0)

            plt.plot([0, ep], [avg_rewards, avg_rewards],
                     label=name_map[method] if method in name_map else method,
                     color=color_map[method] if method in color_map else colors[i])
            plt.fill_between([0, ep], avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color_map[method] if method in color_map else colors[i])
        else:
            plotLearningCurveAvg(rs, window, label=name_map[method] if method in name_map else method,
                                 color=color_map[method] if method in color_map else colors[i],
                                 linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of episodes')
    # if base.find('bbp') > -1:
    plt.ylabel('discounted reward')

    # plt.xlim((-100, ep+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'plot.png'), bbox_inches='tight',pad_inches = 0)

def plotSuccessRate(base, ep=50000, use_default_cm=False, window=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'equi+bufferaug': 'b',
            'cnn+bufferaug': 'g',
            'cnn+rad': 'r',
            'cnn+drq': 'purple',
            'cnn+curl': 'orange',
        }

    linestyle_map = {
    }
    name_map = {
    }

    sequence = {
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/success_rate.npy'))
                if method.find('BC') >= 0 or method.find('tp') >= 0:
                    rs.append(r[-window:].mean())
                else:
                    rs.append(getRewardsSingle(r[:ep], window=window))
            except Exception as e:
                print(e)
                continue

        if method.find('BC') >= 0 or method.find('tp') >= 0:
            avg_rewards = np.mean(rs, axis=0)
            std_rewards = stats.sem(rs, axis=0)

            plt.plot([0, ep], [avg_rewards, avg_rewards],
                     label=name_map[method] if method in name_map else method,
                     color=color_map[method] if method in color_map else colors[i])
            plt.fill_between([0, ep], avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color_map[method] if method in color_map else colors[i])
        else:
            plotLearningCurveAvg(rs, window, label=name_map[method] if method in name_map else method,
                                 color=color_map[method] if method in color_map else colors[i],
                                 linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of episodes')
    # if base.find('bbp') > -1:
    plt.ylabel('success rate')

    # plt.xlim((-100, ep+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'sr.png'), bbox_inches='tight',pad_inches = 0)

def showPerformance(base):
    methods = sorted(filter(lambda x: x[0] != '.', get_immediate_subdirectories(base)))
    for method in methods:
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                rs.append(r[-1000:].mean())
            except Exception as e:
                print(e)
        print('{}: {:.3f}'.format(method, np.mean(rs)))


def plotTDErrors():
    plt.style.use('ggplot')
    colors = "bgrycmkw"
    method_map = {
        'ADET': 'm',
        'ADET+Q*': 'g',
        'DAGGER': 'k',
        'DQN': 'c',
        'DQN+guided': 'y',
        'DQN+Q*': 'b',
        'DQN+Q*+guided': 'r',
        "DQfD": 'chocolate',
        "DQfD+Q*": 'grey'
    }
    i = 0

    base = '/media/dian/hdd/unet/perlin'
    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        if method[0] == '.' or method == 'DAGGER' or method == 'DQN':
            continue
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/td_errors.npy'))
                rs.append(getRewardsSingle(r[:120000], window=1000))
            except Exception as e:
                continue
        if method in method_map:
            plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
        else:
            plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        i += 1

    plt.legend(loc=0)
    plt.xlabel('number of training steps')
    plt.ylabel('TD error')
    plt.yscale('log')
    # plt.ylim((0.8, 0.93))
    plt.show()

def plotLoss(base, step):
    plt.style.use('ggplot')
    colors = "bgrycmkw"
    method_map = {
        'ADET': 'm',
        'ADET+Q*': 'g',
        'DAGGER': 'k',
        'DQN': 'c',
        'DQN+guided': 'y',
        'DQN+Q*': 'b',
        'DQN+Q*+guided': 'r',
        "DQfD": 'chocolate',
        "DQfD+Q*": 'grey'
    }
    i = 0

    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/losses.npy'))[:, 1]
                rs.append(getRewardsSingle(r[:step], window=1000))
            except Exception as e:
                continue
        if method in method_map:
            plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
        else:
            plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        i += 1

    plt.legend(loc=0)
    plt.xlabel('number of training steps')
    plt.ylabel('loss')
    plt.yscale('log')
    # plt.ylim((0.8, 0.93))
    plt.tight_layout()
    plt.savefig(os.path.join(base, 'plot.png'), bbox_inches='tight', pad_inches=0)


def autoDraw(data_dir):
    folder_list = os.listdir(data_dir)
    for folder in folder_list:
        if ".png" in folder:
            pass
        else:
            base = os.path.join(data_dir, folder)
            print(base)
            if ("covid" in base) or ("h4" in base):
                plotEvalCurve(base, 50000, freq=500)
            elif ("corner" in base):
                plotEvalCurve(base, 50000, freq=500)
            else:
                plotEvalCurve(base, 10000, freq=500)

def fixPklFile(base):
    sequence = {
        # for open loop
        'Equi_ASR_SDQfD': '1',
        'CNN_ASR_SDQfD': '2',
        'CNN_RotFCN_SDQfD': '3',
        'CNN_FCN_SDQfD': '4',
        # for close loop
        'CNN_SACfD': '3',
        'Equi_SAC': '2',
        'Equi_SACfD': '1'
    }
    i=0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                # r = np.load(os.path.join(base, method, run, 'info/eval_rewards.npy'))
                data = pickle.load(open(os.path.join(base, method, run, 'log_data.pkl'), 'rb'))
                if (data["num_training_steps"] != 20000) and (data["num_training_steps"] != 50000):
                    temp = data["num_training_steps"]
                    print(f"num_training_steps is {temp}, pls recheck your data, maybe it has been fixed! Exiting!")
                    break
                
                num_training_steps = data["num_training_steps"]//2
                num_eval_intervals = data["num_eval_intervals"]//2
                eval_eps_rewards = data["eval_eps_rewards"][1::2]+[[]]
                eval_eps_dis_rewards = data["eval_eps_dis_rewards"][1::2]+[[]]
                eval_mean_values = data["eval_mean_values"][1::2]+[[]]
                eval_eps_lens = data["eval_eps_lens"][1::2]+[[]]
                new_data = {"num_training_steps":num_training_steps, "num_eval_intervals":num_eval_intervals, 
                            "eval_eps_rewards":eval_eps_rewards, "eval_eps_dis_rewards":eval_eps_dis_rewards,
                            "eval_mean_values":eval_mean_values, "eval_eps_lens":eval_eps_lens}
                with open(os.path.join(base, method, run, 'log_data.pkl'), 'wb') as handle:
                    pickle.dump(new_data, handle)
                
            except Exception as e:
                print(e)
                continue

def autoFix(data_dir):
    folder_list = os.listdir(data_dir)
    for folder in folder_list:
        base = os.path.join(data_dir, folder)
        print(base)
        fixPklFile(base)

def plotEvalCurveNew(base, step=50000, use_default_cm=False, freq=1000, ax=plt):
    # plt.style.use('ggplot')
    # plt.style.use('seaborn-whitegrid')
    # plt.figure(dpi=300)
    # MEDIUM_SIZE = 12
    # BIGGER_SIZE = 14

    # plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            ## for open loop
            'Equi_ASR_SDQfD': 'b',
            'CNN_ASR_SDQfD': 'r',
            'CNN_RotFCN_SDQfD': 'g',
            'CNN_FCN_SDQfD': 'purple',
            ## for close loop
            'Equi_SACfD_PER_Aug': 'b',
            'Equi_SACfD': 'r',
            'CNN_SACfD': 'g',
            'FERM': 'purple',
            'RAD_SACfD': 'orange',
            'DrQ_SACfD': '#a65628',
            # 'Equi_SAC': 'orange',
            ## for open loop 2d
            'equi': 'b',
            'cnn': 'r',
            # for close loop xyz
            'equi_sacfd': 'b',
            'sacfd': 'r',
            'ferm': 'g',
            # open loop 6d
            "Equi_Deictic_ASR": "b",
            "ASR": "r",
            
            
            
        }

    linestyle_map = {
    }
    name_map = {
        # for open loop
        'Equi_ASR_SDQfD': 'Equi ASR',
        'CNN_ASR_SDQfD': 'CNN ASR',
        'CNN_RotFCN_SDQfD': 'Rot FCN',
        'CNN_FCN_SDQfD': 'FCN',
        # for close loop
        'CNN_SACfD': 'SACfD',
        'Equi_SAC': 'Equi SAC',
        'Equi_SACfD': 'Equi SACfD',
        'FERM': 'FERM SACfD',
        'Equi_SACfD_PER_Aug': 'Equi SACfD + PER + Aug',
        'RAD_SACfD': 'RAD SACfD',
        'DrQ_SACfD': 'DrQ SACfD',
        ## for open loop 2d
        'equi': 'Equi FCN',
        'cnn': 'FCN',
        # for close loop xyz
        'equi_sacfd': 'Equi SACfD',
        'sacfd': 'SACfD',
        'ferm': 'FERM SACfD',
        # open loop 6d
        "Equi_Deictic_ASR": "Equi Deictic ASR",
        "ASR": "ASR",

    }

    sequence = {
        # for open loop
        'Equi_ASR_SDQfD': '1',
        'CNN_ASR_SDQfD': '2',
        'CNN_RotFCN_SDQfD': '3',
        'CNN_FCN_SDQfD': '4',
        # for close loop
        'CNN_SACfD': '3',
        # 'Equi_SAC': '3',
        'Equi_SACfD': '2',
        'FERM': '4',
        'Equi_SACfD_PER_Aug': '1',
        'RAD_SACfD': '5',
        'DrQ_SACfD': '6',
        ## for open loop 2d
        'equi': '1',
        'cnn': '2',
        # for close loop xyz
        'equi_sacfd': '1',
        'sacfd': '2',
        'ferm': '3',
        # open loop 6d
        "Equi_Deictic_ASR": "1",
        "ASR": "2",

    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                # r = np.load(os.path.join(base, method, run, 'info/eval_rewards.npy'))
                data = pickle.load(open(os.path.join(base, method, run, 'log_data.pkl'), 'rb'))
                rewards = data['eval_eps_rewards']
                r = [np.mean(x) for x in rewards[:-1]]
                rs.append(r[:step//freq])
            except Exception as e:
                print(e)
                continue

        plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-', ax=ax, legend=False)
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    # ax.legend(loc=0, facecolor='w', fontsize='x-large')
    # ax.xlabel('number of training steps')
    # if base.find('bbp') > -1:
    # ax.ylabel('eval success rate')
    # plt.xlim((-100, step+100))

    # ax.yticks(np.arange(0., 1.05, 0.1))
    # plt.ylim(bottom=-0.05)

    # ax.tight_layout()
    # plt.savefig(os.path.join(base, f'{base}.png'), bbox_inches='tight',pad_inches = 0)

def divideThousand(temp, position):
    return f"{temp/1000}"

# def autoDrawInOne(data_dir):
#     plt.style.use('seaborn-whitegrid')
#     plt.figure(dpi=900)
#     MEDIUM_SIZE = 12
#     BIGGER_SIZE = 14
#     SMALL_SIZE = 8
#     plt.rc('axes', titlesize=10)  # fontsize of the axes title
#     plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
#     plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
#     # plt.tight_layout()
    
#     folder_map = {
#         ## OPEN LOOP
#         "open_stacking": "Block Stacking",
#         "open_4h1": "House Building 1",
#         "open_h2": "House Building 2",
#         "open_h3": "House Building 3",
#         "open_h4": "House Building 4",
#         "open_imh3": "Improvise House Building 3",
#         "open_imh2": "Improvise House Building 2",
#         "open_packing": "Bin Packing",
#         "open_bottle": "Bottle Arrangement",
#         "open_palletizing": "Box Palletizing",
#         "open_covid": "Covid Test",
#         "open_grasping": "Object Grasping",
#         ## CLOSE LOOP
#         "close_reach": "Block Reaching",
#         "close_pick": "Block Picking",
#         "close_push": "Block Pushing",
#         "close_pull": "Block Pulling",
#         "close_bowl": "Block in Bowl",
#         "close_stack": "Block Stacking",
#         "close_2h1": "House Building",
#         "close_corner": "Corner Picking",
#         "close_drawer": "Drawer Opening",
#         "close_clutter_pick": "Object Grasping",

#     }

#     sequence = {
#         ## OPEN LOOP
#         "open_stacking": "01",
#         "open_4h1": "02",
#         "open_h2": "03",
#         "open_h3": "04",
#         "open_h4": "05",
#         "open_imh2": "06",
#         "open_imh3": "07",
#         "open_packing": "08",
#         "open_bottle": "09",
#         "open_palletizing": "10",
#         "open_covid": "11",
#         "open_grasping": "12",
#         ## CLOSE LOOP
#         "close_reach": "01",
#         "close_pick": "02",
#         "close_push": "03",
#         "close_pull": "04",
#         "close_bowl": "05",
#         "close_stack": "06",
#         "close_2h1": "07",
#         "close_corner": "08",
#         "close_drawer": "09",
#         "close_clutter_pick": "10",
#     }

    
#     folder_list = list(filter(lambda x: x[0] != '.', get_immediate_subdirectories(data_dir)))
#     fig, axs = plt.subplots(ceil(len(folder_list)/4), 4, sharey=False, dpi=500, figsize=(10,7))
#     plt.subplots_adjust(top = 0.90, bottom=0.15, right=0.95, left=0.08, hspace=0.4, wspace=0.2)
    
#     i=0
#     for folder in sorted(folder_list, key=lambda x: sequence[x] if x in sequence.keys() else x):
#         base = os.path.join(data_dir, folder)
#         print(base)
#         current_ax=axs[i//4,i%4]
#         if ("covid" in base) or ("h4" in base) or ("corner" in base):
#             plotEvalCurveNew(base, 25000, freq=500, ax=current_ax)
#             current_ax.set(yticks=np.arange(0., 1.05, 0.2), xticks=np.array([0, 10000, 20000, 25000]), xlim=[0,25000])
#         else:
#             plotEvalCurveNew(base, 10000, freq=500, ax=current_ax)
#             current_ax.set(yticks=np.arange(0., 1.05, 0.2), xticks=np.array([0, 2500, 5000, 7500, 10000]), xlim=[0,10000])
#         current_ax.set_title(folder_map[folder],fontweight ="bold")
#         current_ax.xaxis.set_major_formatter(FuncFormatter(divideThousand))
        
#         i+=1
#     fig.supxlabel("Number of training steps (Thousands)", fontweight ="bold", fontsize = 12, y=0.08)
#     fig.supylabel("Eval success rate", fontweight ="bold", fontsize = 12, x=0.03)
#     if "draw_figures_OL" in data_dir:
#         legend = fig.legend(["Equi ASR", "CNN ASR", "Rot FCN", "FCN"], ncol=4,loc="lower center", frameon=True)
#     elif "draw_figures_CL" in data_dir:
#         legend = fig.legend(["Equi SACfD + PER + Aug", "Equi SACfD", "SACfD", "FERM SACfD"], ncol=4,loc="lower center", frameon=True)
#     elif "open_2d" in data_dir:
#         legend = fig.legend(["Equi FCN", "FCN"], ncol=2,loc="lower center", frameon=True)
#     legend.get_frame().set_edgecolor('grey')
#     legend.get_frame().set_linewidth(0.5)
#     legend.get_frame().set_alpha(0.5)
#     fig.savefig(f"{data_dir}.png")

def autoDrawInOne(data_dir, num_column):
    plt.style.use('seaborn-whitegrid')
    plt.figure(dpi=900)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    SMALL_SIZE = 8
    
    # plt.tight_layout()
    
    folder_map = {
        ## OPEN LOOP
        "open_stacking": "Block Stacking",
        "open_4h1": "House Building 1",
        "open_h2": "House Building 2",
        "open_h3": "House Building 3",
        "open_h4": "House Building 4",
        "open_imh3": "Improvise House Building 3",
        "open_imh2": "Improvise House Building 2",
        "open_packing": "Bin Packing",
        "open_bottle": "Bottle Arrangement",
        "open_palletizing": "Box Palletizing",
        "open_covid": "Covid Test",
        "open_grasping": "Object Grasping",
        ## CLOSE LOOP
        "close_reach": "Block Reaching",
        "close_pick": "Block Picking",
        "close_push": "Block Pushing",
        "close_pull": "Block Pulling",
        "close_bowl": "Block in Bowl",
        "close_stack": "Block Stacking",
        "close_2h1": "House Building",
        "close_corner": "Corner Picking",
        "close_drawer": "Drawer Opening",
        "close_clutter_pick": "Object Grasping",
        ## OPEN LOOP 6d
        "ramp_4h1": "Ramp House Building 1",
        "ramp_h2": "Ramp House Building 2",
        "ramp_h3": "Ramp House Building 3",
        "ramp_h4": "Ramp House Building 4",
        "ramp_4s": "Ramp Block Stacking",
        "ramp_imh2": "Ramp Improvise House Building 2",
        "ramp_imh3": "Ramp Improvise House Building 3",
        "bump_h4": "Bump House Building 4",
        "bump_box": "Bump Box Palletizing",
        ## OPEN LOOP 2d
        "open2d_stacking": "Block Stacking",
        "open2d_4h1": "House Building 1",
        "open2d_h2": "House Building 2",
        "open2d_h3": "House Building 3",
        "open2d_h4": "House Building 4",
        "open2d_imh3": "Improvise House Building 3",
        "open2d_imh2": "Improvise House Building 2",
        "open2d_packing": "Bin Packing",
        "open2d_bottle": "Bottle Arrangement",
        "open2d_palletizing": "Box Palletizing",
        "open2d_covid": "Covid Test",
        "open2d_grasping": "Object Grasping",

    }

    sequence = {
        ## OPEN LOOP
        "open_stacking": "01",
        "open_4h1": "02",
        "open_h2": "03",
        "open_h3": "04",
        "open_h4": "05",
        "open_imh2": "06",
        "open_imh3": "07",
        "open_packing": "08",
        "open_bottle": "09",
        "open_palletizing": "10",
        "open_covid": "11",
        "open_grasping": "12",
        ## CLOSE LOOP
        "close_reach": "01",
        "close_pick": "02",
        "close_push": "03",
        "close_pull": "04",
        "close_bowl": "05",
        "close_stack": "06",
        "close_2h1": "07",
        "close_corner": "08",
        "close_drawer": "09",
        "close_clutter_pick": "10",
        ## OPEN LOOP 6d
        "ramp_4h1": "02",
        "ramp_h2": "03",
        "ramp_h3": "04",
        "ramp_h4": "05",
        "ramp_4s": "01",
        "ramp_imh2": "06",
        "ramp_imh3": "07",
        "bump_h4": "08",
        "bump_box": "09",
    }

    if num_column == 4:
        title_factor = 1
        small_factor = 1
        figure_size_x_factor=1
        figure_size_y_factor=1
    if num_column == 3:
        title_factor = 0.7
        small_factor = 0.8
        figure_size_x_factor = 0.625
        figure_size_y_factor = 0.75
    plt.rc('axes', titlesize=10*title_factor)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE*small_factor)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE*small_factor)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE*small_factor)  # fontsize of the tick labels

    folder_list = list(filter(lambda x: x[0] != '.', get_immediate_subdirectories(data_dir)))
    fig, axs = plt.subplots(ceil(len(folder_list)/num_column), num_column, sharey=False, dpi=500, figsize=(10*figure_size_x_factor,7*figure_size_y_factor))
    plt.subplots_adjust(top = 0.90, bottom=0.15, right=0.95, left=0.08, hspace=0.4, wspace=0.2)
    
    i=0
    for folder in sorted(folder_list, key=lambda x: sequence[x] if x in sequence.keys() else x):
        base = os.path.join(data_dir, folder)
        print(base)
        if (num_column == 4) and (len(folder_list) + 2 == np.ceil(len(folder_list)/4)*4) and (i//4 == np.ceil(len(folder_list)/4)-1):
            current_ax=axs[i//4,i%4+1]    
        else:
            current_ax=axs[i//num_column,i%num_column]
        if "open_6d_new" in base:
            plotEvalCurveNew(base, 25000, freq=500, ax=current_ax)
            current_ax.set(yticks=np.arange(0., 1.05, 0.2), xticks=np.array([0, 10000, 20000, 25000]), xlim=[0,25000], ylim=[0,1.05])
        elif ("covid" in base) or ("h4" in base):
            plotEvalCurveNew(base, 25000, freq=500, ax=current_ax)
            current_ax.set(yticks=np.arange(0., 1.05, 0.2), xticks=np.array([0, 10000, 20000, 25000]), xlim=[0,25000], ylim=[0,1.05])
        elif ("corner" in base):
            plotEvalCurveNew(base, 20000, freq=500, ax=current_ax)
            current_ax.set(yticks=np.arange(0., 1.05, 0.2), xticks=np.array([0, 5000, 10000, 15000, 20000]), xlim=[0,20000], ylim=[0,1.05])
        else:
            plotEvalCurveNew(base, 10000, freq=500, ax=current_ax)
            current_ax.set(yticks=np.arange(0., 1.05, 0.2), xticks=np.array([0, 2500, 5000, 7500, 10000]), xlim=[0,10000], ylim=[0,1.05])
        current_ax.set_title(folder_map[folder],fontweight ="bold")
        current_ax.xaxis.set_major_formatter(FuncFormatter(divideThousand))
        
        i+=1
    fig.supxlabel("Number of training steps (Thousands)", fontweight ="bold", fontsize = 12*small_factor, y=0.08)
    fig.supylabel("Eval success rate", fontweight ="bold", fontsize = 12*small_factor, x=0.03)
    if ("draw_figures_OL" in data_dir):
        legend = fig.legend(["Equi ASR", "CNN ASR", "Rot FCN", "FCN"], ncol=4,loc="lower center", frameon=True)
    elif ("draw_figures_CL" in data_dir) or ("001NEW_CL" in data_dir):
        legend = fig.legend(["Equi SACfD + PER + Aug", "Equi SACfD", "SACfD", "FERM SACfD"], ncol=4,loc="lower center", frameon=True)
    elif "open_2d" in data_dir:
        legend = fig.legend(["Equi FCN", "FCN"], ncol=2,loc="lower center", frameon=True)
    elif "close_xyz" in data_dir:
        legend = fig.legend(["Equi SACfD", "SACfD", "FERM SACfD"], ncol=3,loc="lower center", frameon=True)
    elif "open_6d" in data_dir:
        legend = fig.legend(["Equi Deictic ASR", "ASR"], ncol=2,loc="lower center", frameon=True)
    elif "CL_extra" in data_dir:
        legend = fig.legend(["Equi SACfD + PER + Aug", "Equi SACfD", "SACfD", "FERM SACfD", "RAD SACfD", "DrQ SACfD"], ncol=6,loc="lower center", frameon=True)
    elif "OPEN" in data_dir or "CLOSE" in data_dir:
        legend = fig.legend(["Equi"], ncol=1,loc="lower center", frameon=True)
    elif "001NEW_OL_extra" in data_dir:
        legend = fig.legend(["SDQfD", "DQfD", "ADET", "DQN"], ncol=4,loc="lower center", frameon=True)
    legend.get_frame().set_edgecolor('grey')
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_alpha(0.5)
    if (num_column == 4) and (len(folder_list) + 2 == np.ceil(len(folder_list)/4)*4):
        fig.delaxes(axs[ceil(len(folder_list)/4)-1,0])
        fig.delaxes(axs[ceil(len(folder_list)/4)-1,3])
    fig.savefig(f"{data_dir}.png")
if __name__ == '__main__':
    base = '/home/mingxi/ws/001results/env_result/draw_figures_OL/open_4h1'
    # plotLearningCurve(base, 1000, window=20)
    # plotSuccessRate(base, 3000, window=100)

    # plotEvalCurve(base, 10000, freq=500) # rl_env

    # showPerformance(base)
    # plotLoss(base, 30000)

    # plotStepSRCurve(base, 10000, freq=500, file_name='step_success_rate') # 
    # plotStepRewardCurve(base, 10000, freq=200)


    # # auto figure genereator for env paper
    data_dir = "/home/huy/Documents/Robotics/BulletArm/draw_figures_OL"
    # autoDraw(data_dir)
    autoDrawInOne(data_dir, num_column=4) 
    ## draw one figure for open loop
    # drawOL(data_dir)

    #---------Env paper fix-----------#
    # fix one file
    # base = '/home/mingxi/ws/001results/env_result/draw_figures_CL/close_2h1'
    # fixPklFile(base)

    # fix files in one folder
    # data_dir = '/home/mingxi/ws/001results/env_result/draw_figures_CL'
    # autoFix(data_dir)
