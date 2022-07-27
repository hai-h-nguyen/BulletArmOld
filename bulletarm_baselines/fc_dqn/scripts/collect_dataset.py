from datetime import datetime
import numpy as np
from bulletarm import env_factory
import bulletarm.envs.configs as env_configs
from bulletarm_baselines.fc_dqn.utils.dataset import ListDataset, count_objects, decompose_objects

import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def create_folder(path):
    try:
        os.mkdir(path)
    except:
        print(f'[INFO] folder {path} existed, can not create new')

def create_envs_(num_processes=10, env_config=env_configs.DEFAULT_CONFIG, planner_config={"pos_noise": 0.0, "rot_noise": 0.0, "random_orientation": True}):
    env_config['num_processes'] = num_processes
    return env_factory.createEnvs(num_processes, "house_building_x_deconstruct", env_config,
            planner_config
        )

def collect(goal_str= "1b1b1r", num_samples=10000, debug=False):
        '''
        Gather deconstruction transitions and reverse them for construction

        Args:
        - goal_str: the goal task
        - num_sample: The number of sample in dataset
        - debug: checking collect data

        Returns: list of transitions. Each transition is in the form of
        ((state, in_hand, obs), action, reward, done, (next_state, next_in_hand, next_obs))
        '''
        num_objects = count_objects(goal_str)
        num_classes = 2 * num_objects - 1
        num_blocks, num_bricks, num_triangles, num_roofs = decompose_objects(goal_str)

        print("=================================")
        print("Collect data: {:s} goal, {:d} objects".format(goal_str, num_objects))
        print(f"Num blocks {num_blocks}, Num bricks {num_bricks}, Num triangles {num_triangles}, Num roofs {num_roofs}")
        print("=================================")


        config = env_configs.DEFAULT_CONFIG
        config['goal_string'] = goal_str
        env = create_envs_(env_config=config)

        dataset = ListDataset()

        num_episodes = num_samples // num_classes
        
        if debug:
            obss = []
            inhands = []
            labels = []
            states = []
            num_episodes = 1

        transitions = env.gatherDeconstructTransitions(num_episodes)
        env.close()
        transitions.reverse()
        print(len(transitions))
        true_index = [i for i in range(len(transitions)) if transitions[i][3] is True]
        perfect_index = [true_index[i] for i in range(len(true_index)) if (true_index[0] == num_classes-2) or (true_index[i]-true_index[i-1] == num_classes-1)]
        print(len(true_index))
        print(len(perfect_index))
        # exit()

        for i in perfect_index:
            for j in range(num_classes-1, 0, -1):
                if debug:
                    states.append(transitions[i-j+1][0][0])
                    obss.append(transitions[i-j+1][0][2])
                    inhands.append(transitions[i-j+1][0][1])
                    labels.append(j)

                dataset.add("HAND_BITS", transitions[i-j+1][0][0])
                dataset.add("OBS", transitions[i-j+1][0][2])
                dataset.add("HAND_OBS", transitions[i-j+1][0][1])
                dataset.add("DONES", j)
                dataset.add("ABS_STATE_INDEX", j)
                    
                if j == 1:
                    if debug:
                        states.append(transitions[i][4][0])
                        obss.append(transitions[i][4][2])
                        inhands.append(transitions[i][4][1])
                        labels.append(0)
                    dataset.add("HAND_BITS", transitions[i][4][0])
                    dataset.add("OBS", transitions[i][4][2])
                    dataset.add("HAND_OBS", transitions[i][4][1])
                    dataset.add("DONES", 1)
                    dataset.add("ABS_STATE_INDEX", 0)

        if debug:
            create_folder('check_debug_collect_image')
            for i in range(len(states)):
                plt.figure(figsize=(15,4))
                plt.subplot(1,2,1)
                plt.imshow(obss[i], cmap='gray')
                plt.colorbar()

                plt.subplot(1,2,2)
                plt.imshow(inhands[i], cmap='gray')
                plt.colorbar()
                plt.suptitle(f"Label: {labels[i]}, State: {states[i]}")
                plt.savefig(f'check_debug_collect_image/image_{i}.png')
                
        dataset = dataset.to_array_dataset({
            "HAND_BITS": np.int32, "OBS": np.float32, "HAND_OBS": np.float32,
            "DONES": np.bool,
            "ABS_STATE_INDEX": np.int32,
        })
        dataset.metadata = {
            "NUM_EXP": dataset.size, "TIMESTAMP": str(datetime.today())
        }
        print(dataset.size)
        dataset.save_hdf5(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.h5")

        print("DONE!!!")

if __name__ == '__main__':
    collect(goal_str='2b1l2r', num_samples=20000, debug=True)
