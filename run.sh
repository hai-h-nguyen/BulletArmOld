# export PYTHONPATH=/home/hnguyen/long_branch/huy_new/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hnguyen/long_branch/house/BulletArm/:$PYTHONPATH
export PYTHONPATH=/home/hainguyen/long_branch/BulletArm/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=2 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
 --algorithm=dqn --architecture=equi_asr --env=house_building_3 --fill_buffer_deconstruct\
 --planner_episode=100 --max_train_step=10000 --wandb_group=goal_100 --device_name=cuda\
 --wandb_logs=1 --use_classifier=0 --get_bad_pred=0 --seed=0 --classifier_name=perfect\
 --use_proser=0 --use_equivariant=1 --dummy_number=1


# -------------- collect data -------------- #
# python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py

# -----------train classifier -------------- #
# python bulletarm_baselines/fc_dqn/scripts/all_about_classifier.py -gs block_stacking -ufm False -up False

# python multi_run.py