# export PYTHONPATH=/home/hnguyen/long_branch/huy_new/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hnguyen/long_branch/house/BulletArm/:$PYTHONPATH
export PYTHONPATH=/home/huy/Documents/Robotics/BulletArm/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python bulletarm_baselines/fc_dqn/scripts/main_goal.py \
--algorithm=sdqfd --architecture=equi_asr --env=house_building_1 --fill_buffer_deconstruct \
--planner_episode=5 --max_train_step=10000 --wandb_group=SDQfD_ASR_classifier_goal_5 --wandb_seed=s1 \
--device_name=cuda --wandb_logs=1 --use_classifier=0 --get_bad_pred=0 --seed=1


# -------------- collect data -------------- #
# python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py

# -----------train classifier -------------- #
# python bulletarm_baselines/fc_dqn/scripts/all_about_classifier.py -gs house_building_1 -ufm False -up False

# python multi_run.py