from bulletarm_baselines.fc_dqn.utils.SoftmaxClassifier import SoftmaxClassifier, SupConEmbedding, LinearClassifier
from bulletarm_baselines.fc_dqn.utils.View import View
from bulletarm_baselines.fc_dqn.utils.ConvEncoder import ConvEncoder, CNNOBSEncoder, CNNHandObsEncoder
from bulletarm_baselines.fc_dqn.utils.SplitConcat import SplitConcat
from bulletarm_baselines.fc_dqn.utils.FCEncoder import FCEncoder
from bulletarm_baselines.fc_dqn.utils.EquiObs import EquiObs
from bulletarm_baselines.fc_dqn.utils.EquiHandObs import EquiHandObs
from bulletarm_baselines.fc_dqn.utils.dataset import ArrayDataset, count_objects
from bulletarm_baselines.fc_dqn.utils.result import Result
from bulletarm_baselines.fc_dqn.utils.SupConlosses import SupConLoss

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, accuracy_score, classification_report

def create_folder(path):
    try:
        os.mkdir(path)
    except:
        print(f'[INFO] folder {path} existed, can not create new')

def load_dataset(goal_str, validation_fraction=0.75, eval=False):
    dataset = ArrayDataset(None)
    if eval:
        print(f"=================\t Loading eval dataset {goal_str} \t=================")
        dataset.load_hdf5(f"/home/hnguyen/huy/final/BulletArm/bulletarm_baselines/fc_dqn/classifiers/fail_data_{goal_str}_goal_25_dqn_normal.h5")
        num_samples = dataset.size
        print(f"Total number samples: {num_samples}")
        abs_index = dataset["TRUE_ABS_STATE_INDEX"]
        print(f"Class: {np.unique(abs_index, return_counts=True)[0]}")
        print(f"Number samples/each class: {np.unique(abs_index, return_counts=True)[1]}")
        return dataset
    else:
        print(f"=================\t Loading training dataset {goal_str}\t=================")
        dataset.load_hdf5(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.h5")
        dataset.shuffle()
        num_samples = dataset.size
        print(f"Total number samples: {num_samples}")
        abs_index = dataset["ABS_STATE_INDEX"]
        print(f"Class: {np.unique(abs_index, return_counts=True)[0]}")
        print(f"Number samples/each class: {np.unique(abs_index, return_counts=True)[1]}")
        valid_samples = int(num_samples * validation_fraction)
        valid_dataset = dataset.split(valid_samples)

        dataset.size = dataset.size - valid_dataset.size
        return dataset, valid_dataset


class State_abstractor():
    def __init__(self, goal_str=None, use_equivariant=None, device=None):
        self.goal_str = goal_str
        self.use_equivariant = use_equivariant
        self.device = device
        self.batch_size = 128

        if self.goal_str == 'block_stacking':
            num_objects = 4
        elif self.goal_str == 'house_building_1':
            num_objects = 4
        elif self.goal_str == 'house_building_2':
            num_objects = 3
        elif self.goal_str == 'house_building_3':
            num_objects = 4
        elif self.goal_str == 'house_building_4':
            num_objects = 6
        else:
            num_objects = count_objects(self.goal_str)
        self.num_classes = 2 * num_objects - 1

        if self.use_equivariant:
            self.name = 'equi_' + self.goal_str
        else:
            self.name = self.goal_str

        self.build_state_abstractor()

    def build_state_abstractor(self):

        if self.use_equivariant:
            print('='*50)
            print('----------\t Equivaraint Model \t -----------')
            print('='*50)
            conv_obs = EquiObs(num_subgroups=4)
            conv_hand_obs = EquiHandObs(num_subgroups=4)
        else:    
            conv_obs = CNNOBSEncoder()
            conv_hand_obs = CNNHandObsEncoder()

        conv_obs_view = View([128])
        conv_obs_encoder = nn.Sequential(conv_obs, conv_obs_view)

        conv_hand_obs_view = View([128])
        conv_hand_obs_encoder = nn.Sequential(conv_hand_obs, conv_hand_obs_view)

        conv_encoder = SplitConcat([conv_obs_encoder, conv_hand_obs_encoder], 1)

        intermediate_fc = FCEncoder({
            "input_size": 256,
            "neurons": [256, 128],
            "use_batch_norm": True,
            "use_layer_norm": False,
            "activation_last": True
        })

        encoder = nn.Sequential(conv_encoder, intermediate_fc, nn.Dropout(p=0.1))

        encoder.output_size = 128

        self.classifier = SoftmaxClassifier(encoder, self.num_classes, conv_encoder)
        self.classifier.to(self.device)
        return self.classifier

    def train_state_abstractor(self, num_training_steps=10000, learning_rate=1e-3, weight_decay=1e-5, visualize=True):
        self.classifier.train()
        # Load dataset
        self.dataset, self.valid_dataset = load_dataset(goal_str=self.goal_str)
        epoch_size = len(self.dataset['OBS']) // self.batch_size
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)}')
        opt = optim.Adam(self.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_val_loss, best_classifier = None, None
        best_step = None

        result = Result()
        result.register("TOTAL_LOSS")
        result.register("ACCURACY")
        result.register("TOTAL_VALID_LOSS")
        result.register("VALID_ACCURACY")
        
        for training_step in range(num_training_steps):
            epoch_step = training_step % epoch_size
            if epoch_step == 0:
                self.dataset.shuffle()
            
            if training_step % 500 == 0:
                valid_loss, valid_acc = self.validate(dataset=self.valid_dataset)
                if best_val_loss is None or best_val_loss > valid_loss:
                    best_val_loss = valid_loss
                    best_step = training_step
                    best_classifier = cp.deepcopy(self.classifier.state_dict())
                    print(f'step {training_step}: best = {best_val_loss} at {best_step}')
                result.add("TOTAL_VALID_LOSS", valid_loss)
                result.add("VALID_ACCURACY", valid_acc)
                print("validation complete")
            
            if training_step % 100 == 0:
                print("step {:d}".format(training_step))
            
            opt.zero_grad()
            obs, hand_obs, abs_task_indices = self.get_batch(epoch_step=epoch_step, dataset=self.dataset)
            loss, acc = self.classifier.compute_loss_and_accuracy([obs, hand_obs], abs_task_indices)
            loss.backward()
            opt.step()
            result.add_pytorch("TOTAL_LOSS", loss)
            result.add("ACCURACY", acc)

        if best_classifier is not None:
            self.classifier.load_state_dict(best_classifier)
        else:
            print("Best model not saved.")

        self.classifier.eval()
        final_valid_loss = self.validate(dataset=self.valid_dataset)
        print(f"Best Valid Loss: {final_valid_loss[0]} and Best Valid Accuracy: {final_valid_loss[1]}")
        # torch.save(self.classifier.state_dict(), f"bulletarm_baselines/fc_dqn/classifiers/{self.name}.pt")   

        self.find_mean_cov()

        if visualize:
            self.plot_result(result=result)
            self.plot_TSNE(embedding=self.classifier.embedding)

    def validate(self, dataset):
        self.classifier.eval()
        result = Result()
        result.register("TOTAL_LOSS")
        result.register("ACCURACY")
        # throws away a bit of data if validation set size % batch size != 0
        num_steps = int(len(dataset["OBS"]) // self.batch_size)
        for step in range(num_steps):
            obs, hand_obs, abs_task_indices = self.get_batch(epoch_step=step, dataset=dataset)
            loss, acc = self.classifier.compute_loss_and_accuracy([obs, hand_obs], abs_task_indices)
            result.add_pytorch("TOTAL_LOSS", loss)
            result.add("ACCURACY", acc)
        self.classifier.train()
        return result.mean("TOTAL_LOSS"), result.mean("ACCURACY")

    def get_batch(self, epoch_step, dataset):
        b = np.index_exp[epoch_step * self.batch_size: (epoch_step + 1) * self.batch_size]

        obs = dataset["OBS"][b]
        hand_obs = dataset["HAND_OBS"][b]
        abs_state_index = dataset["ABS_STATE_INDEX"][b]

        return torch.from_numpy(obs[:, np.newaxis, :, :]).to(self.device), \
            torch.from_numpy(hand_obs[:, np.newaxis, :, :]).to(self.device), \
            torch.from_numpy(abs_state_index).to(self.device)

    def plot_result(self, result):
        losses = np.stack(result["TOTAL_LOSS"], axis=0)
        valid_losses = np.stack(result["TOTAL_VALID_LOSS"], axis=0)
        acc = np.stack(result["ACCURACY"], axis=0)
        valid_acc = np.stack(result["VALID_ACCURACY"], axis=0)

        # Plot Loss and Acc curve
        create_folder('Loss_and_Acc')
        plt.figure(figsize=(20, 20))
        x = np.arange(0, valid_losses.shape[0])
        x *= 500

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

        plt.savefig(f'Loss_and_Acc/{self.name}.png')
        plt.close()

    def plot_TSNE(self, embedding, state="before"):
        embedding.eval()

        out_train = []
        label_train = []
        for i in range(self.dataset.size):
            obs = torch.from_numpy(self.dataset['OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            hand_obs = torch.from_numpy(self.dataset['HAND_OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            out_train.append(embedding([obs, hand_obs]).detach().cpu().numpy().reshape(256))
            label_train.append(self.dataset['ABS_STATE_INDEX'][i])

        out_train = np.array(out_train)
        label_train = np.array(label_train)

        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne_train = tsne.fit_transform(out_train)

        df = pd.DataFrame()
        df["label"] = label_train
        df["dim_1"] = tsne_train[:, 0]
        df["dim_2"] = tsne_train[:, 1]
        plt.figure(figsize=(20, 20))
        sns.scatterplot(x="dim_1", y="dim_2", hue=df.label.tolist(), palette=sns.color_palette("hls", self.num_classes), data=df, s=10).set(title=f"TSNE of {self.name}")
        plt.savefig(f'TSNE/Training_{self.name}_TSNE_{state}.png')
        plt.close()

        out_val = []
        label_val = []
        for i in range(self.valid_dataset.size):
            obs = torch.from_numpy(self.valid_dataset['OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            hand_obs = torch.from_numpy(self.valid_dataset['HAND_OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            out_val.append(embedding([obs, hand_obs]).detach().cpu().numpy().reshape(256))
            label_val.append(self.valid_dataset['ABS_STATE_INDEX'][i])

        load_outlier = load_dataset(goal_str=self.goal_str, eval=True)
        for i in range(load_outlier.size):
            if load_outlier['TRUE_ABS_STATE_INDEX'][i] == self.num_classes:
                obs = torch.from_numpy(load_outlier['OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
                hand_obs = torch.from_numpy(load_outlier['HAND_OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
                out_val.append(embedding([obs, hand_obs]).detach().cpu().numpy().reshape(256))
                label_val.append(load_outlier['TRUE_ABS_STATE_INDEX'][i])
        
        out_val = np.array(out_val)
        label_val = np.array(label_val)

        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne_val = tsne.fit_transform(out_val)

        df = pd.DataFrame()
        df["label"] = label_val
        df["dim_1"] = tsne_val[:, 0]
        df["dim_2"] = tsne_val[:, 1]

        plt.figure(figsize=(20, 20))
        sns.scatterplot(x="dim_1", y="dim_2", hue=df.label.tolist(), palette=sns.color_palette("hls", self.num_classes+1), data=df, s=10).set(title=f"TSNE of {self.name}")
        plt.savefig(f'TSNE/Validation_{self.name}_TSNE_{state}.png')
        plt.close()

    def compute_mahalanobis(self, mu, inv_sigma, x):
        return np.sqrt(np.dot(np.dot((x - mu).T, inv_sigma), (x - mu)))

    def find_mean_cov(self):
        self.classifier.eval()
        out = {}
        for i in range(self.num_classes):
            features = []
            idx = np.where(self.dataset['ABS_STATE_INDEX'] == i)[0]
            for j in idx:
                obs = torch.from_numpy(self.dataset['OBS'][j][np.newaxis, np.newaxis, :, :]).to(self.device)
                hand_obs = torch.from_numpy(self.dataset['HAND_OBS'][j][np.newaxis, np.newaxis, :, :]).to(self.device)
                features.append(self.classifier.embedding([obs, hand_obs]).detach().cpu().numpy().reshape(256))
            feature_list = cp.deepcopy(features)
            features = np.array(features)
            mu = np.mean(features, axis=0)
            sigma = np.cov(features.T)
            pinv_sigma = np.linalg.pinv(sigma)
            d = []
            for feature in feature_list:
                d.append(self.compute_mahalanobis(mu, pinv_sigma, feature))
            d = np.array(d)
            d = np.quantile(d, 0.95)
            out[i] = [mu, pinv_sigma, d*1.25]

        
        true_label = []
        pred_label = []
        for i in range(self.valid_dataset.size):
            true_label.append(self.valid_dataset['ABS_STATE_INDEX'][i])
            obs = torch.from_numpy(self.valid_dataset['OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            hand_obs = torch.from_numpy(self.valid_dataset['HAND_OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            feature = self.classifier.embedding([obs, hand_obs]).detach().cpu().numpy().reshape(256)
            d = []
            for j in range(self.num_classes):
                mu, inv_sigma, _ = out[j]
                dist = self.compute_mahalanobis(mu, inv_sigma, feature)
                d.append(dist)
            d_min = min(d)
            index_d = d.index(d_min)
            if d_min > out[index_d][2]:
                pred_label.append(self.num_classes)
            else:
                pred_label.append(self.classifier.get_prediction(x=[obs, hand_obs], hard=True).detach().cpu().numpy().reshape(1)[0])
        acc = accuracy_score(true_label, pred_label)
        print(f'Accuracy: {acc}')
        
        outlier_dataset = load_dataset(goal_str=self.goal_str, eval=True)
        pred_label = []
        fp = 0
        for i in range(outlier_dataset.size):
            obs = torch.from_numpy(outlier_dataset['OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            hand_obs = torch.from_numpy(outlier_dataset['HAND_OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            feature = self.classifier.embedding([obs, hand_obs]).detach().cpu().numpy().reshape(256)
            # calculate mahanalobis distance
            d = []
            for j in range(self.num_classes):
                mu, inv_sigma, _ = out[j]
                dist = self.compute_mahalanobis(mu, inv_sigma, feature)
            d.append(dist)
            d_min = min(d)
            index_d = d.index(d_min)
            if d_min > out[index_d][2]:
                pred_label.append(self.num_classes)
            else:
                l = self.classifier.get_prediction(x=[obs, hand_obs], hard=True).detach().cpu().numpy().reshape(1)[0]
                if l == outlier_dataset['TRUE_ABS_STATE_INDEX'][i]:
                    fp += 1
        print(f'Outlier Acc: {len(pred_label)}')
        print(f'Correct False Positive: {fp}')

    def load_classifier(self):
        self.classifier.train()
        self.classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/{self.name}.pt"))
        self.classifier.eval()
        print(f'------\t Successfully load classifier {self.name}\t-----------')
        return self.classifier

class SupCon_State_abstractor(State_abstractor):
    def __init__(self, goal_str, use_equivariant=False, device=torch.device('cuda')):
        self.goal_str = goal_str
        self.use_equivariant = use_equivariant
        self.device = device
        self.batch_size = 128

        if self.goal_str == 'block_stacking':
            num_objects = 4
        elif self.goal_str == 'house_building_1':
            num_objects = 4
        elif self.goal_str == 'house_building_2':
            num_objects = 3
        elif self.goal_str == 'house_building_3':
            num_objects = 4
        elif self.goal_str == 'house_building_4':
            num_objects = 6
        else:
            num_objects = count_objects(self.goal_str)
        self.num_classes = 2 * num_objects - 1

        if self.use_equivariant:
            self.name = 'equi_' + self.goal_str
        else:
            self.name = self.goal_str
        self.name = '00supcon_' + self.name

        self.transform1 = transforms.Compose([
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.RandomRotation(180),
            transforms.GaussianBlur((5, 9), (0.1, 2.0)),
        ])

        self.transform2 = transforms.Compose([
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.GaussianBlur((5, 9), (0.1, 2.0)),
        ])

        self.build_embedding()

    def build_embedding(self):
        if self.use_equivariant:
            print('='*50)
            print('----------\t Equivaraint Model \t -----------')
            print('='*50)
            conv_obs = EquiObs(num_subgroups=4)
            conv_hand_obs = EquiHandObs(num_subgroups=4)
        else:    
            conv_obs = CNNOBSEncoder()
            conv_hand_obs = CNNHandObsEncoder()

        conv_obs_view = View([128])
        conv_obs_encoder = nn.Sequential(conv_obs, conv_obs_view)

        conv_hand_obs_view = View([128])
        conv_hand_obs_encoder = nn.Sequential(conv_hand_obs, conv_hand_obs_view)

        conv_encoder = SplitConcat([conv_obs_encoder, conv_hand_obs_encoder], 1)
        conv_encoder.output_size = 256
        self.embedding = SupConEmbedding(conv_encoder)
        self.embedding.to(self.device)
        return self.embedding

    def train_embedding(self, num_training_steps=10000, visualize=True):
        self.embedding.train()
        criterion = SupConLoss(temperature=0.1)
        optimizer = torch.optim.Adam(self.embedding.parameters(), lr=1e-3, weight_decay=1e-5)

        self.dataset, self.valid_dataset = load_dataset(goal_str=self.goal_str)
        self.epoch_size = len(self.dataset['OBS']) // self.batch_size

        best_loss = np.inf
        best_embedding = None
        losses = []
        
        for training_step in range(num_training_steps):
            epoch_step = training_step % self.epoch_size
            if epoch_step == 0:
                self.dataset.shuffle()

            if training_step % 100 == 0:
                print(f'Step {training_step}')

            optimizer.zero_grad()
            obs, hand_obs, abs_task_indices = self.get_batch(epoch_step=epoch_step, dataset=self.dataset)
            bsz = obs.shape[0]
            obs1 = self.transform1(obs)
            obs2 = self.transform1(obs)
            hand_obs1 = self.transform2(hand_obs)
            hand_obs2 = self.transform2(hand_obs)

            obs_cat = torch.cat([obs1, obs2], dim=0)
            hand_obs_cat = torch.cat([hand_obs1, hand_obs2], dim=0)

            feature = self.embedding([obs_cat, hand_obs_cat])
            f1, f2 = torch.split(feature, [bsz, bsz], dim=0)
            feature = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(feature, abs_task_indices)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_loss = loss
                best_embedding = cp.deepcopy(self.embedding.state_dict())

        if best_embedding is not None:
            self.embedding.load_state_dict(best_embedding)
        else:
            print("Best embedding not found")
        self.embedding.eval()
        if visualize:
            plt.figure(figsize=(20, 20))
            plt.plot(losses)
            plt.xlabel('Training steps')
            plt.ylabel('Loss')
            plt.title('Training loss')
            plt.savefig(f'Loss_and_Acc/{self.name}.png')
            plt.close()

            self.plot_TSNE(embedding=self.embedding.encoder)

    def train_linear_classifier(self, num_training_steps=10000):
        self.embedding.eval()
        self.cls = LinearClassifier(input_dim=self.embedding.encoder.output_size, num_classes=self.num_classes)
        self.cls.to(self.device)
        self.cls.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cls.parameters(), lr=1e-3)

        best_loss = np.inf
        best_cls = None

        for training_step in range(num_training_steps):
            epoch_step =  training_step % self.epoch_size
            if epoch_step == 0:
                self.dataset.shuffle()

            if training_step % 500 == 0:
                valid_loss, valid_acc = self.validate()
                if valid_loss < best_loss:
                    print("Save best classifier")
                    best_loss = valid_loss
                    best_cls = cp.deepcopy(self.cls.state_dict())
                    print(f'step {training_step}: best loss: {best_loss} and best acc: {valid_acc}')

            if training_step % 100 == 0:
                print(f'step {training_step}')

            optimizer.zero_grad()
            obs, hand_obs, abs_task_indices = self.get_batch(epoch_step=epoch_step, dataset=self.dataset)
            with torch.no_grad():
                feature = self.embedding.encoder([obs, hand_obs])
            pred = self.cls(feature)
            loss = criterion(pred, abs_task_indices.long())
            loss.backward()
            optimizer.step()

        if best_cls is not None:
            self.cls.load_state_dict(best_cls)
        else:
            print("Best classifier not found")

        self.find_mean_cov()         
    
    def validate(self):
        self.embedding.eval()
        self.cls.eval()
        result = Result()
        result.register("TOTAL_LOSS")
        result.register("ACCURACY")
        num_steps = int(len(self.valid_dataset['OBS']) // self.batch_size)
        for i in range(num_steps):
            obs, hand_obs, abs_task_indices = self.get_batch(epoch_step=i, dataset=self.valid_dataset)
            with torch.no_grad():
                feature = self.embedding.encoder([obs, hand_obs])
            loss, acc = self.cls.compute_loss_and_accuracy(feature, abs_task_indices)
            result.add_pytorch("TOTAL_LOSS", loss)
            result.add("ACCURACY", acc)
        self.cls.train()
        return result.mean("TOTAL_LOSS"), result.mean("ACCURACY")

    def find_mean_cov(self):
        self.embedding.eval()
        self.cls.eval()

        out = {}
        for i in range(self.num_classes):
            features = []
            idx = np.where(self.dataset['ABS_STATE_INDEX'] == i)[0]
            for j in idx:
                obs = torch.from_numpy(self.dataset['OBS'][j][np.newaxis, np.newaxis, :, :]).to(self.device)
                hand_obs = torch.from_numpy(self.dataset['HAND_OBS'][j][np.newaxis, np.newaxis, :, :]).to(self.device)
                with torch.no_grad():
                    f = self.embedding.encoder([obs, hand_obs])
                features.append(f.detach().cpu().numpy().reshape(256))
            feature_list = cp.deepcopy(features)
            features = np.array(features)
            mu = np.mean(features, axis=0)
            sigma = np.cov(features.T)
            pinv_sigma = np.linalg.pinv(sigma)
            d = []
            for f in feature_list:
                d.append(self.compute_mahalanobis(mu, pinv_sigma, f))
            d = np.array(d)
            d = np.quantile(d, 0.95)
            out[i] = [mu, pinv_sigma, d*1.25]

        true_label = []
        pred_label = []
        for i in range(self.valid_dataset.size):
            obs = torch.from_numpy(self.valid_dataset['OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            hand_obs = torch.from_numpy(self.valid_dataset['HAND_OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            true_label.append(self.valid_dataset['ABS_STATE_INDEX'][i])
            with torch.no_grad():
                f = self.embedding.encoder([obs, hand_obs])
            fe = f.detach().cpu().numpy().reshape(256)
            d = []
            for j in range(self.num_classes):
                mu, inv_sigma, _ = out[j]
                d.append(self.compute_mahalanobis(mu, inv_sigma, fe))
            d_min = min(d)
            index_d = d.index(d_min)
            if d_min > out[index_d][2]:
                pred_label.append(self.num_classes)
            else:
                l = self.cls(f).argmax().detach().cpu().numpy()
                pred_label.append(l)
        acc = accuracy_score(true_label, pred_label)
        print(f'Accuracy: {acc}')

        outlier_dataset = load_dataset(goal_str=self.goal_str, eval=True)
        pred_label = []
        fp = 0
        for i in range(outlier_dataset.size):
            obs = torch.from_numpy(outlier_dataset['OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            hand_obs = torch.from_numpy(outlier_dataset['HAND_OBS'][i][np.newaxis, np.newaxis, :, :]).to(self.device)
            with torch.no_grad():
                f = self.embedding.encoder([obs, hand_obs])
            fe = f.detach().cpu().numpy().reshape(256)
            d = []
            for j in range(self.num_classes):
                mu, inv_sigma, _ = out[j]
                d.append(self.compute_mahalanobis(mu, inv_sigma, fe))
            d_min = min(d)
            index_d = d.index(d_min)
            if d_min > out[index_d][2]:
                pred_label.append(self.num_classes)
            else:
                l = self.cls(f).argmax().detach().cpu().numpy()
                if l == outlier_dataset['TRUE_ABS_STATE_INDEX'][i]:
                    fp += 1
        print(f'Outlier: {len(pred_label)}')
        print(f'Correct false positive: {fp}')

if __name__ == '__main__':
    # Build argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_str', type=str, default='house_building_4', help='Goal string')
    parser.add_argument('--use_equivariant', type=bool, default=False, help='Use equivariant model')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device to use')
    parser.add_argument('--supcon', type=bool, default=False, help='Use supcon model')
    parser.add_argument('--visualize', type=bool, default=False, help='Visualize training')
    args = parser.parse_args()

    if args.supcon:
        model = SupCon_State_abstractor(goal_str=args.goal_str, use_equivariant=args.use_equivariant, device=torch.device(args.device))
        model.train_embedding(num_training_steps=1500, visualize=args)
        model.train_linear_classifier(num_training_steps=1500)
    else:
        model = State_abstractor(goal_str=args.goal_str, use_equivariant=args.use_equivariant, device=torch.device(args.device))
        model.train_state_abstractor(num_training_steps=15000, visualize=args.visualize)
