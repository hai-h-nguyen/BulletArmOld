# export PYTHONPATH=/home/hnguyen/long_branch/huy_new/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hnguyen/long_branch/house/BulletArm/:$PYTHONPATH
export PYTHONPATH=/home/hnguyen/huy/final/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hainguyen/long_branch/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/huy/Documents/Robotics/BulletArm/:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=2 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
#  --algorithm=dqn --architecture=equi_asr --env=house_building_1 --fill_buffer_deconstruct\
#  --planner_episode=5 --max_train_step=10000 --wandb_group=goal_5 --device_name=cuda\
#  --wandb_logs=0 --use_classifier=1 --get_bad_pred=1 --seed=0 --classifier_name=normal\
#  --use_equivariant=0 --dummy_number=1
CUDA_VISIBLE_DEVICES=3 python bulletarm_baselines/fc_dqn/scripts/main_hqdn.py\
 --algorithm=dqn --architecture=equi_asr --env=block_stacking --fill_buffer_deconstruct\
 --planner_episode=5 --max_train_step=10000 --wandb_group=goal_5 --device_name=cuda\
 --wandb_logs=0 --use_classifier=0 --get_bad_pred=0 --seed=0 --classifier_name=normal\
 --use_equivariant=0 --dummy_number=1 --num_eps_meta=3000

# -------------- collect data -------------- #
# python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py

# -----------train classifier -------------- #
# CUDA_VISIBLE_DEVICES=1 python bulletarm_baselines/fc_dqn/scripts/State_abstractor.py

# python multi_run.py --file=cmd1.yaml