# export PYTHONPATH=/home/hnguyen/long_branch/huy_new/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hnguyen/long_branch/house/BulletArm/:$PYTHONPATH
export PYTHONPATH=/home/hnguyen/huy/BulletArm/:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 python bulletarm_baselines/fc_dqn/scripts/main_goal.py \
# --algorithm=dqn --architecture=equi_asr --env=house_building_2 --fill_buffer_deconstruct \
# --planner_episode=5 --max_train_step=10000 --wandb_group=goal_5 \
# --device_name=cuda --wandb_logs=1 --use_classifier=1 --get_bad_pred=1 --seed=0 \
# --classifier_name=equi_proser

# -------------- collect data -------------- #
# python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py

# -----------train classifier -------------- #
# python bulletarm_baselines/fc_dqn/scripts/all_about_classifier.py -gs house_building_1 -ufm False -up False

python multi_run.py