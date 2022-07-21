export PYTHONPATH=/home/huy/Desktop/RL-research/BulletArm/:$PYTHONPATH
python bulletarm_baselines/fc_dqn/scripts/main_goal.py --algorithm=dqn \
--architecture=equi_asr --env=block_stacking --fill_buffer_deconstruct --planner_episode=300
# python bulletarm_baselines/fc_dqn/scripts/main_goal.py
