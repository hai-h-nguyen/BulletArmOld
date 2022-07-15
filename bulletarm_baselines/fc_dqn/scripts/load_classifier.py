from bulletarm_baselines.fc_dqn.utils.SoftmaxClassifier import SoftmaxClassifier
from bulletarm_baselines.fc_dqn.utils.View import View
from bulletarm_baselines.fc_dqn.utils.ConvEncoder import ConvEncoder
from bulletarm_baselines.fc_dqn.utils.SplitConcat import SplitConcat
from bulletarm_baselines.fc_dqn.utils.FCEncoder import FCEncoder

import torch
import torch.nn as nn


def load_classifier(goal_str):

    # encodes obs of shape Bx1x90x90 into Bx128x5x5
    conv_obs = ConvEncoder({
        "input_size": [90, 90, 1],
        "filter_size": [3, 3, 3, 3],
        "filter_counts": [32, 64, 128, 128],
        "strides": [2, 2, 2, 2],
        "use_batch_norm": True,
        "activation_last": True,
        "flat_output": False
    })
    # average pool Bx128x5x5 into Bx128x1x1 and reshape that into Bx128
    conv_obs_avg_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
    conv_obs_view = View([128])
    conv_obs_encoder = nn.Sequential(conv_obs, conv_obs_avg_pool, conv_obs_view)

    # encodes hand obs of shape Bx1x24x24 into Bx128x1x1
    conv_hand_obs = ConvEncoder({
        "input_size": [24, 24, 1],
        "filter_size": [3, 3, 3, 3],
        "filter_counts": [32, 64, 128, 128],
        "strides": [2, 2, 2, 2],
        "use_batch_norm": True,
        "activation_last": True,
        "flat_output": False
    })
    # reshape Bx128x1x1 into Bx128
    conv_hand_obs_view = View([128])
    conv_hand_obs_encoder = nn.Sequential(conv_hand_obs, conv_hand_obs_view)
    # gets [obs, hand_obs], runs that through their respective encoders
    # and then concats [Bx128, Bx128] into Bx256
    conv_encoder = SplitConcat([conv_obs_encoder, conv_hand_obs_encoder], 1)

    fc = FCEncoder({
        "input_size": 256,
        "neurons": [256, 256],
        "use_batch_norm": True,
        "use_layer_norm": False,
        "activation_last": True
    })

    encoder = nn.Sequential(conv_encoder, fc)

    encoder.output_size = 256

    # TODO: Replace 5 with the actual number of classes
    classifier = SoftmaxClassifier(encoder, 5)
    classifier.to("cuda")
    classifier.eval()

    classifier.load_state_dict(torch.load(f"../classifiers/{goal_str}.pt"))

    return classifier


if __name__ == "__main__":
    load_classifier(goal_str='1b1r')