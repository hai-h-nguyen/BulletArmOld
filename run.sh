# export PYTHONPATH=/home/hnguyen/long_branch/huy_new/BulletArm/:$PYTHONPATH
export PYTHONPATH=/home/hnguyen/huy/BulletArm/:$PYTHONPATH
python bulletarm_baselines/fc_dqn/scripts/main_goal.py --algorithm=dqn \
--architecture=equi_asr --env=house_building_2 --fill_buffer_deconstruct \
 --planner_episode=5 --max_train_step=10000 --wandb_group=DQN_ASR_goal_5 --wandb_seed=s3 --device_name=cuda:2 --wandb_logs=1

# -------------- collect data -------------- #
# python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py

# -----------train classifier -------------- #
# python bulletarm_baselines/fc_dqn/scripts/all_about_classifier.py

# python tests/test_bullet_house_3.py