export PYTHONPATH=/home/huy/Desktop/RL-research/BulletArm/:$PYTHONPATH
# python bulletarm_baselines/fc_dqn/scripts/main_goal.py --algorithm=dqn \
# --architecture=equi_asr --env=house_building_3 --fill_buffer_deconstruct \
#  --planner_episode=200

# -------------- collect data -------------- #
# python bulletarm_baselines/fc_dqn/scripts/collect_dataset.py

# -----------train classifier -------------- #
# python bulletarm_baselines/fc_dqn/scripts/all_about_classifier.py

python tests/test_bullet_house_3.py