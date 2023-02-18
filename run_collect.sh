# export PYTHONPATH=/home/hnguyen/long_branch/huy_new/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hnguyen/long_branch/house/BulletArm/:$PYTHONPATH
export PYTHONPATH=/home/hnguyen/huy/final/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hainguyen/long_branch/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/huy/Documents/Robotics/BulletArm/:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=2 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
#  --algorithm=dqn --architecture=equi_asr --env=house_building_1 --fill_buffer_deconstruct\
#  --planner_episode=0 --max_train_step=10000 --wandb_group=goal_5 --device_name=cuda\
#  --wandb_logs=0 --use_classifier=1 --get_bad_pred=1 --seed=0 --classifier_name=normal\
#  --use_equivariant=0 --dummy_number=1

#  CUDA_VISIBLE_DEVICES=2 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
#  --algorithm=dqn --architecture=equi_asr --env=house_building_2 --fill_buffer_deconstruct\
#  --planner_episode=5 --max_train_step=10000 --wandb_group=goal_5 --device_name=cuda\
#  --wandb_logs=0 --use_classifier=1 --get_bad_pred=1 --seed=0 --classifier_name=normal\
#  --use_equivariant=0 --dummy_number=1

#  CUDA_VISIBLE_DEVICES=2 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
#  --algorithm=dqn --architecture=equi_asr --env=house_building_2 --fill_buffer_deconstruct\
#  --planner_episode=5 --max_train_step=25000 --wandb_group=goal_5 --device_name=cuda\
#  --wandb_logs=0 --use_classifier=1 --get_bad_pred=1 --seed=0 --classifier_name=normal\
#  --use_equivariant=0 --dummy_number=1

 CUDA_VISIBLE_DEVICES=2 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
 --algorithm=dqn --architecture=equi_asr --env=1l2b2r --fill_buffer_deconstruct\
 --planner_episode=5 --max_train_step=25000 --wandb_group=goal_5 --device_name=cuda\
 --wandb_logs=0 --use_classifier=1 --get_bad_pred=1000 --seed=0 --classifier_name=normal\
 --use_equivariant=0 --dummy_number=1

#  CUDA_VISIBLE_DEVICES=2 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
#  --algorithm=dqn --architecture=equi_asr --env=house_building_1 --fill_buffer_deconstruct\
#  --planner_episode=5 --max_train_step=25000 --wandb_group=goal_5 --device_name=cuda\
#  --wandb_logs=0 --use_classifier=1 --get_bad_pred=1000 --seed=0 --classifier_name=normal\
#  --use_equivariant=0 --dummy_number=1

#  CUDA_VISIBLE_DEVICES=2 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
#  --algorithm=dqn --architecture=equi_asr --env=house_building_2 --fill_buffer_deconstruct\
#  --planner_episode=5 --max_train_step=25000 --wandb_group=goal_5 --device_name=cuda\
#  --wandb_logs=0 --use_classifier=1 --get_bad_pred=1000 --seed=0 --classifier_name=normal\
#  --use_equivariant=0 --dummy_number=1

 CUDA_VISIBLE_DEVICES=2 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
 --algorithm=dqn --architecture=equi_asr --env=house_building_4 --fill_buffer_deconstruct\
 --planner_episode=25 --max_train_step=25000 --wandb_group=goal_25 --device_name=cuda\
 --wandb_logs=0 --use_classifier=1 --get_bad_pred=2000 --seed=0 --classifier_name=normal\
 --use_equivariant=0 --dummy_number=1
