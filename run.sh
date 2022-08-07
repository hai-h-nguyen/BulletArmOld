export PYTHONPATH=/home/hnguyen/long_branch/huy_new/BulletArm/:$PYTHONPATH
python bulletarm_baselines/fc_dqn/scripts/main_goal.py --algorithm=dqn \
--architecture=equi_asr --env=house_building_1 --fill_buffer_deconstruct \
 --planner_episode=15 --max_train_step=30000

# -------------- collect data -------------- #
# python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py

# -----------train classifier -------------- #
# python bulletarm_baselines/fc_dqn/scripts/all_about_classifier.py

# python tests/test_bullet_house_3.py