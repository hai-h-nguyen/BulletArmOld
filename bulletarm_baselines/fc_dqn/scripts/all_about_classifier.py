from ast import arg
from cProfile import label
from cgi import test

from matplotlib import use
from bulletarm_baselines.fc_dqn.utils.SoftmaxClassifier import SoftmaxClassifier
from bulletarm_baselines.fc_dqn.utils.View import View
from bulletarm_baselines.fc_dqn.utils.ConvEncoder import ConvEncoder
from bulletarm_baselines.fc_dqn.utils.SplitConcat import SplitConcat
from bulletarm_baselines.fc_dqn.utils.FCEncoder import FCEncoder
from bulletarm_baselines.fc_dqn.utils.EquiConv import EquiConv
from bulletarm_baselines.fc_dqn.utils.dataset import ArrayDataset, count_objects, decompose_objects
from bulletarm_baselines.fc_dqn.utils.result import Result

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import argparse

def create_folder(path):
    try:
        os.mkdir(path)
    except:
        print(f'[INFO] folder {path} existed, can not create new')

def load_dataset(goal_str, validation_fraction=0.2):
    dataset = ArrayDataset(None)
    dataset.load_hdf5(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.h5")
    dataset.shuffle()
    num_samples = dataset.size
    print("Loading dataset")
    print(f"Total number samples: {num_samples}")
    abs_index = dataset["ABS_STATE_INDEX"]
    print(f"Class: {np.unique(abs_index, return_counts=True)[0]}")
    print(f"Number samples/each class: {np.unique(abs_index, return_counts=True)[1]}")
    valid_samples = int(num_samples * validation_fraction)
    valid_dataset = dataset.split(valid_samples)

    test_dataset = dataset.split(100)
    return dataset, valid_dataset, test_dataset


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

def build_optimizer(classifier=None, learning_rate=1e-3, weight_decay=1e-4):
    params = classifier.parameters()
    print("num parameter tensors: {:d}".format(len(list(classifier.parameters()))))
    opt = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    return opt

def get_batch(epoch_step, batch_size, validation, dataset, valid_dataset, device):
    b = np.index_exp[epoch_step * batch_size: (epoch_step + 1) * batch_size]

    if validation:
        obs = valid_dataset["OBS"][b]
        hand_obs = valid_dataset["HAND_OBS"][b]
        abs_state_index = valid_dataset["ABS_STATE_INDEX"][b]
    else:
        obs = dataset["OBS"][b]
        hand_obs = dataset["HAND_OBS"][b]
        abs_state_index = dataset["ABS_STATE_INDEX"][b]
    return torch.from_numpy(obs[:, np.newaxis, :, :]).to(device), \
        torch.from_numpy(hand_obs[:, np.newaxis, :, :]).to(device), \
        torch.from_numpy(abs_state_index).to(device)

def train_classifier(goal_str, use_equivariant, num_training_steps, batch_size, device, learning_rate, weight_decay):
    num_objects = count_objects(goal_str)
    num_classes = 2 * num_objects - 1
    num_blocks, num_bricks, num_triangles, num_roofs = decompose_objects(goal_str)
    print("=================================")
    print("Training classifier for task: {:s} goal, {:d} objects".format(goal_str, num_objects))
    print(f"Num blocks {num_blocks}, Num bricks {num_bricks}, Num triangles {num_triangles}, Num roofs {num_roofs}")
    print("=================================")

    dataset, valid_dataset, test_dataset = load_dataset(goal_str=goal_str)
    epoch_size = dataset["OBS"].shape[0] // batch_size


    classifier = build_classifier(num_classes=num_classes, use_equivariant=use_equivariant)
    classifier.train()

    opt = build_optimizer(classifier=classifier, learning_rate=learning_rate, weight_decay=weight_decay)

    best_val_loss, best_classifier = None, None

    result = Result()
    result.register("TOTAL_LOSS")
    result.register("ACCURACY")
    result.register("TOTAL_VALID_LOSS")
    result.register("VALID_ACCURACY")

    for training_step in range(num_training_steps):
        epoch_step = training_step % epoch_size
        if epoch_step == 0:
            dataset.shuffle()
        if training_step % 300 == 0:
            valid_loss, valid_acc = validate(classifier=classifier, dataset=dataset, valid_dataset=valid_dataset, batch_size=batch_size, device=device)
            if best_val_loss is None or best_val_loss > valid_loss:
                best_val_loss = valid_loss
                best_classifier = cp.deepcopy(classifier.state_dict())
            result.add("TOTAL_VALID_LOSS", valid_loss)
            result.add("VALID_ACCURACY", valid_acc)
            print("validation complete")
        if training_step % 100 == 0:
            print("step {:d}".format(training_step))
        obs, hand_obs, abs_task_indices = get_batch(epoch_step=epoch_step, batch_size=batch_size, dataset=dataset, valid_dataset=valid_dataset, device=device, validation=False)
        opt.zero_grad()
        loss, acc = classifier.compute_loss_and_accuracy([obs, hand_obs], abs_task_indices)
        loss.backward()
        opt.step()
        result.add_pytorch("TOTAL_LOSS", loss)
        result.add("ACCURACY", acc)

    if best_classifier is not None:
        classifier.load_state_dict(best_classifier)
    else:
        print("Best model not saved.")
    losses = np.stack(result["TOTAL_LOSS"], axis=0)
    valid_losses = np.stack(result["TOTAL_VALID_LOSS"], axis=0)
    acc = np.stack(result["ACCURACY"], axis=0)
    valid_acc = np.stack(result["VALID_ACCURACY"], axis=0)
    create_folder('Loss_and_Acc')
    plt.figure(figsize=(8, 6))
    x = np.arange(0, valid_losses.shape[0])
    x *= 300
    
    plt.subplot(3, 1, 1)
    plt.plot(losses, linestyle='-', color='blue', label="Training loss")
    plt.plot(x, valid_losses, linestyle='--', color='red', marker='*', label='Valid loss')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(losses, linestyle='-', color='blue', label="Training loss (log)")
    plt.plot(x, valid_losses, linestyle='--', color='red', marker='*', label='Valid loss (log)')
    plt.yscale('log')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(acc, linestyle='-', color='blue', label='Training acc')
    plt.plot(x, valid_acc, linestyle='--', color='red', marker='*', label='Valid acc')
    plt.legend()

    plt.savefig(f'Loss_and_Acc/loss_acc_curve_{goal_str}.png')

    classifier.eval()
    final_valid_loss = validate(classifier=classifier, dataset=dataset, valid_dataset=valid_dataset, batch_size=batch_size, device=device)
    print(f" Valid Loss: {final_valid_loss[0]} and Valid Accuracy: {final_valid_loss[1]}")
    test_loss = validate(classifier=classifier, dataset=dataset, valid_dataset=test_dataset, batch_size=batch_size, device=device)
    print(f"Test Loss: {test_loss[0]} and Test Accuracy: {test_loss[1]}")
    torch.save(classifier.state_dict(), f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.pt")


def validate(classifier, dataset, valid_dataset, batch_size, device):
    classifier.eval()
    result = Result()
    result.register("TOTAL_LOSS")
    result.register("ACCURACY")
    # throws away a bit of data if validation set size % batch size != 0
    num_steps = int(len(valid_dataset["OBS"]) // batch_size)
    for step in range(num_steps):
        obs, hand_obs, abs_task_indices = get_batch(epoch_step=step, batch_size=batch_size, dataset=dataset, valid_dataset=valid_dataset, device=device, validation=True)
        loss, acc = classifier.compute_loss_and_accuracy([obs, hand_obs], abs_task_indices)
        result.add_pytorch("TOTAL_LOSS", loss)
        result.add("ACCURACY", acc)
    classifier.train()
    return result.mean("TOTAL_LOSS"), result.mean("ACCURACY")

def finetune_model_to_proser(goal_str, use_equivariant, batch_size, device, dummy_number, finetune_epoch, finetune_learning_rate, lamda0, lamda1, lamda2):
    num_objects = count_objects(goal_str)
    num_classes = 2 * num_objects - 1
    dataset, valid_dataset = load_dataset(goal_str=goal_str)
    epoch_size = dataset["OBS"].shape[0] // batch_size

    classifier = load_classifier(goal_str=goal_str, use_equivariant=use_equivariant)
    classifier.create_dummy(dummy_number=dummy_number)
    classifier.cuda()

    finetune_loss = nn.CrossEntropyLoss()
    finetune_optimizer = optim.Adam(classifier.parameters(), lr=finetune_learning_rate)
    scheduler = optim.lr_scheduler.StepLR(finetune_optimizer, step_size=5, gamma=0.5)

    dataset_check = ArrayDataset(None)
    dataset_check.load_hdf5(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}_check.h5")
    num_check_samples = dataset_check.size
    print("Loading dataset check")
    print(f"Total number samples: {num_check_samples}")
    abs_index_check = dataset_check["ABS_STATE_INDEX"]
    print(f"Class: {np.unique(abs_index_check, return_counts=True)[0]}")
    print(f"Number samples/each class: {np.unique(abs_index_check, return_counts=True)[1]}")

    false_id = []
   
    print(f"Starting finetune model with dummy class: {dummy_number} and number epoch: {finetune_epoch}")
    count_false_best = 0
    for fi_ep in range(finetune_epoch):
        dataset.shuffle()
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        percent = []
        for i in range(epoch_size):
            obs, hand_obs, abs_task_indices = get_batch(epoch_step=i, batch_size=batch_size, dataset=dataset, valid_dataset=valid_dataset, device=device, validation=False)
            finetune_optimizer.zero_grad()
            beta = torch.distributions.Beta(1, 1).sample([]).item()

            halflength = int(len(obs)/2)

            prehalf_obs = obs[:halflength]
            prehalf_hand_obs = hand_obs[:halflength]
            prehalf_label = abs_task_indices[:halflength]
            posthalf_obs = obs[halflength:]
            posthalf_hand_obs = hand_obs[halflength:]
            poshalf_label = abs_task_indices[halflength:]
            index = torch.randperm(prehalf_obs.size(0)).to(device)
            pre2embeddings = classifier.pre2block([prehalf_obs, prehalf_hand_obs])
            mixed_embeddings = beta*pre2embeddings + (1-beta)*pre2embeddings[index]

            dummylogit = classifier.dummypredict([posthalf_obs, posthalf_hand_obs])
            post_outputs = classifier.forward([posthalf_obs, posthalf_hand_obs])
            posthalf_output = torch.cat((post_outputs, dummylogit), 1)
            prehalf_output = torch.cat((classifier.latter2blockclf1(mixed_embeddings), classifier.latter2blockclf2(mixed_embeddings)), 1)
            maxdummy, _ = torch.max(dummylogit.clone(), dim=1)
            maxdummy = maxdummy.view(-1, 1)
            dummyoutputs = torch.cat((post_outputs.clone(), maxdummy), dim=1)
            for i in range(len(dummyoutputs)):
                nowlabel = poshalf_label[i]
                dummyoutputs[i][nowlabel] = -1e-9
            dummytargets = torch.ones_like(poshalf_label)*num_classes
            outputs = torch.cat((prehalf_output, posthalf_output), 0)
            loss1 = finetune_loss(prehalf_output, (torch.ones_like(prehalf_label)*num_classes).long().to(device))
            loss2 = finetune_loss(posthalf_output, poshalf_label.long())
            loss3 = finetune_loss(dummyoutputs, dummytargets.long())
            loss = lamda0 * loss1 +  lamda1 * loss2 + lamda2 * loss3
            loss.backward()
            finetune_optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += abs_task_indices.size(0)
            correct += predicted.eq(abs_task_indices).sum().item()
            percent.append(correct/total)
        percent = np.array(percent)
        print(f"Finetune Epoch {fi_ep}: {percent.mean()}")

        best_finetune_model, count_false_best = validate_finetune_model(classifier=classifier, dataset_check=dataset_check, false_id=false_id, num_classes=num_classes, min_false_count=count_false_best, device=device)
        scheduler.step()

    classifier.load_state_dict(best_finetune_model)
    final_valid_loss_finetune = validate(classifier=classifier, dataset=dataset, valid_dataset=valid_dataset, batch_size=batch_size, device=device)
    print(f"Loss: {final_valid_loss_finetune[0]} and Accuracy: {final_valid_loss_finetune[1]} with validate dataset")

def validate_finetune_model(classifier, dataset_check, false_id, num_classes, min_false_count, device):
    classifier.eval()
    
    false_count = 0
    known_count = 0
    sum_pre_false = 0
    for i in range(dataset_check.size):
        obs = torch.from_numpy(dataset_check["OBS"][i][np.newaxis, np.newaxis, :, :]).to(device)
        hand_obs = torch.from_numpy(dataset_check["HAND_OBS"][i][np.newaxis, np.newaxis, :, :]).to(device)
        pre = classifier.proser_prediction([obs, hand_obs])
        
        if pre == num_classes:
            sum_pre_false += 1
        if i+1 in false_id:
            if pre == num_classes:
                false_count += 1
        else:
            if pre == dataset_check["ABS_STATE_INDEX"][i]:
                known_count += 1
    print(f'Number of true predict with known classes/ Number of true sample: {known_count}/{dataset_check.size-len(false_id)} = {known_count * 100/(dataset_check.size-len(false_id))}')
    print(f'Number of true false predict / Number of false sample: {false_count}/{len(false_id)} = {false_count*100/len(false_id)}')
    print(f'Number of true false predict / Number of predict false sample: {false_count}/{sum_pre_false} = {false_count * 100 / sum_pre_false}')
    
    if known_count/(dataset_check.size-len(false_id)) > 0.95 and false_count > min_false_count:
        best_finetune_model = cp.deepcopy(classifier.state_dict())
        return best_finetune_model, false_count
    return cp.deepcopy(classifier.state_dict()), min_false_count
    

def load_classifier(goal_str, use_equivariant):
    num_objects = count_objects(goal_str)
    num_classes = 2 * num_objects - 1
    classifier = build_classifier(num_classes=num_classes, use_equivariant=use_equivariant)
    classifier.eval()
    classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.pt"))
    print('------\t Successfully load classifier \t-----------')
    return classifier


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-gs', '--goal_str', default='2b1l2r', help='The goal string task')
    ap.add_argument('-bs', '--batch_size', default=64, help='Number of samples in a batch')
    ap.add_argument('-nts', '--num_training_steps', default=10000, help='Number of training step')
    ap.add_argument('-dv', '--device', default='cuda', help='Having gpu or not')
    ap.add_argument('-lr', '--learning_rate', default=1e-3, help='Learning rate')
    ap.add_argument('-wd', '--weight_decay', default=1e-5, help='Weight decay')
    ap.add_argument('-ufm', '--use_equivariant', default=False, help='Using equivariant or not')
    ap.add_argument('-up', '--use_proser', default=False, help='Using Proser (open-set recognition) or not')
    ap.add_argument('-dn', '--dummy_number', default=5, help='Number of dummy classifiers')
    ap.add_argument('-fep', '--finetune_epoch', default=30, help='Number of finetune epoch')
    ap.add_argument('-ld0', '--lamda0', default=0.01, help='Weight for data placeholder loss')
    ap.add_argument('-ld1', '--lamda1', default=1, help='Weight for classifier placeholder loss (mapping the nearest to ground truth label)')
    ap.add_argument('-ld2', '--lamda2', default=1, help='Weight for classifier placeholder loss (mapping the second nearest to the dummpy classifier )')


    args = vars(ap.parse_args())

    if not args['use_proser']:
        train_classifier(goal_str=args['goal_str'], use_equivariant=args['use_equivariant'], num_training_steps=args['num_training_steps'], batch_size=args['batch_size'], device=args['device'], learning_rate=args['learning_rate'], weight_decay=args['weight_decay'])
    else:
        finetune_model_to_proser(goal_str=args['goal_str'], use_equivariant=args['use_equivariant'], batch_size=args['batch_size'], device=args['device'], dummy_number=args['dummy_number'], finetune_epoch=args['finetune_epoch'], finetune_learning_rate=args['learning_rate'], lamda0=args['lamda0'], lamda1=args['lamda1'], lamda2=args['lamda2'])
