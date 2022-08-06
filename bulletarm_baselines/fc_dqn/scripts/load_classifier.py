from bulletarm_baselines.fc_dqn.utils.SoftmaxClassifier import SoftmaxClassifier
from bulletarm_baselines.fc_dqn.utils.View import View
from bulletarm_baselines.fc_dqn.utils.ConvEncoder import ConvEncoder
from bulletarm_baselines.fc_dqn.utils.SplitConcat import SplitConcat
from bulletarm_baselines.fc_dqn.utils.FCEncoder import FCEncoder
from bulletarm_baselines.fc_dqn.utils.EquiConv import EquiConv
from bulletarm_baselines.fc_dqn.utils.dataset import ArrayDataset, count_objects
from bulletarm_baselines.fc_dqn.utils.result import Result
import torch
import torch.nn as nn
from bulletarm_baselines.fc_dqn.utils.parameters import *

class block_stacking_perfect_classifier(nn.Module):
  def __init__(self):

    super(block_stacking_perfect_classifier, self).__init__()
  
  def check_equal(self, a ,b):
    return abs(a-b)<0.001

  def forward(self,obs,inhand):
    len = obs.shape[0]
    res = []
    for i in range(len):
        obs_height = torch.max(obs[i])
        in_hand_height = torch.max(inhand[i])
        if (not (self.check_equal(in_hand_height,0) or self.check_equal(in_hand_height,0.03))):
            in_hand_height = torch.tensor(0.03)

        if (self.check_equal(obs_height,0.03) and self.check_equal(in_hand_height,0)):
            res.append(6)
            continue
        if (self.check_equal(obs_height,0.03) and self.check_equal(in_hand_height,0.03)):
            res.append(5)
            continue
        if (self.check_equal(obs_height,0.06) and self.check_equal(in_hand_height,0)):
            res.append(4)
            continue
        if (self.check_equal(obs_height,0.06) and self.check_equal(in_hand_height,0.03)):
            res.append(3)
            continue
        if (self.check_equal(obs_height,0.09) and self.check_equal(in_hand_height,0)):
            res.append(2)
            continue
        if (self.check_equal(obs_height,0.09) and self.check_equal(in_hand_height,0.03)):
            res.append(1)
            continue
        if (self.check_equal(obs_height,0.12) and self.check_equal(in_hand_height,0)):
            res.append(0)
            continue
        res.append(6)
        # raise NotImplementedError(f'error classifier with obs_height = {obs_height}, in_hand_height = {in_hand_height}')
        
    return torch.tensor(res).to('cuda')

def build_classifier(num_classes, use_equivariant=False):
    """
    Build model classifier

    Args:
    - num_classes
    """

    # encodes obs of shape Bx1x128x128 into Bx128x5x5
    if use_equivariant:
        print('===========================')
        print('----------\t Equivaraint Model \t -----------')
        print('===========================')
        conv_obs = EquiConv(num_subgroups=4, filter_sizes=[3, 3, 3, 3, 3, 3], filter_counts=[32, 64, 128, 256, 256, 128])
        conv_obs_avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

    else:    
        conv_obs = ConvEncoder({
            "input_size": [128, 128, 1],
            "filter_size": [3, 3, 3, 3, 3],
            "filter_counts": [32, 64, 128, 256, 128],
            "strides": [2, 2, 2, 2, 2],
            "use_batch_norm": True,
            "activation_last": True,
            "flat_output": False
        })
        # average pool Bx128x5x5 into Bx128x1x1 and reshape that into Bx128
        conv_obs_avg_pool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
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

    intermediate_fc = FCEncoder({
        "input_size": 256,
        "neurons": [256, 256],
        "use_batch_norm": True,
        "use_layer_norm": False,
        "activation_last": True
    })

    encoder = nn.Sequential(conv_encoder, intermediate_fc, nn.Dropout(p=0.3))

    encoder.output_size = 256

    classifier = SoftmaxClassifier(encoder, conv_encoder, intermediate_fc, num_classes)
    classifier.to("cuda")
    return classifier

def load_classifier(goal_str, num_classes, use_equivariant=False, use_proser=False, dummy_number=1):
    classifier = build_classifier(num_classes=num_classes, use_equivariant=use_equivariant)
    classifier.train()
    # if use_proser:
    #     classifier.create_dummy(dummy_number=dummy_number)
    #     if use_equivariant:
    #         classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/finetune_equi_{goal_string}.pt"))
    #     else:
    #         classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/finetune_{goal_string}.pt"))
    # else:
    #     if use_equivariant:
    #         classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/equi_{goal_str}.pt"))
    #     else:
    #         classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.pt"))
    classifier.to(device)
    classifier.eval()
    print('------\t Successfully load classifier \t-----------')
    return classifier

  
if __name__ == "__main__":
    load_classifier(goal_str='1b1r',use_equivariant=False)