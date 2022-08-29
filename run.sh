# export PYTHONPATH=/home/hnguyen/long_branch/huy_new/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hnguyen/long_branch/house/BulletArm/:$PYTHONPATH
export PYTHONPATH=/home/hainguyen/long_branch/BulletArm/:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=1 
python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
 --algorithm=dqn --architecture=equi_asr --env=house_building_4 --fill_buffer_deconstruct\
 --planner_episode=10 --max_train_step=25000 --wandb_group=goal_10 --device_name=cuda:1\
 --wandb_logs=1 --use_classifier=1 --get_bad_pred=0 --seed=4 --classifier_name=equi\
 --use_proser=0 --use_equivariant=1 --dummy_number=1

# -------------- collect data -------------- #
python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py

# -----------train classifier -------------- #
# python bulletarm_baselines/fc_dqn/scripts/all_about_classifier.py --goal_str=1l2b1r --use_equivariant=False --use_proser=False

# python multi_run.py