import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class SoftmaxClassifier(nn.Module):

    def __init__(self, encoder, num_classes):

        super(SoftmaxClassifier, self).__init__()

        self.encoder = encoder
        self.num_classes = num_classes

        self.create_fc_()
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.fc(self.encoder(x))
    

    def get_prediction(self, x, logits=False, hard=False):

        pred = self.forward(x)

        if not logits:
            pred = self.softmax(pred)

        if hard:
            pred = torch.argmax(pred, dim=1)

        return pred

    def compute_loss_and_accuracy(self, x, y):

        pred = self.forward(x)
        loss = self.loss(input=pred, target=y.long())
        # loss = self.loss(pred, y.long())

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        acc_np = np.mean(np.equal(np.argmax(pred_np, axis=1).astype(np.int32), y_np.astype(np.int32)).astype(np.float32))

        return torch.mean(loss), acc_np

    def create_fc_(self):

        self.fc = nn.Linear(self.encoder.output_size, self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0)

class SupConEmbedding(nn.Module):
    def __init__(self, encoder, head='mlp', feature_dim=128):
        super(SupConEmbedding, self).__init__()
        self.encoder = encoder
        
        if head == 'linear':
            self.head = nn.Linear(self.encoder.output_size, feature_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.encoder.output_size, self.encoder.output_size),
                nn.GELU(),
                nn.Linear(self.encoder.output_size, feature_dim)
            )
        else:
            raise ValueError('Unsupported head: {}'.format(head))
        self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(self.head(x), dim=1)
        return x
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class LinearClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):

        super(LinearClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.create_fc_()
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.fc(x)
    

    def get_prediction(self, x, logits=False, hard=False):

        pred = self.forward(x)

        if not logits:
            pred = self.softmax(pred)

        if hard:
            pred = torch.argmax(pred, dim=1)

        return pred

    def compute_loss_and_accuracy(self, x, y):

        pred = self.forward(x)
        loss = self.loss(input=pred, target=y.long())
        # loss = self.loss(pred, y.long())

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        acc_np = np.mean(np.equal(np.argmax(pred_np, axis=1).astype(np.int32), y_np.astype(np.int32)).astype(np.float32))

        return torch.mean(loss), acc_np

    def create_fc_(self):

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim//2, bias=True),
            nn.GELU(),
            nn.Linear(self.input_dim//2, self.num_classes, bias=True),
        )
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
