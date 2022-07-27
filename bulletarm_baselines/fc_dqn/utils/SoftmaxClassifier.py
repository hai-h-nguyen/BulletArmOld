import numpy as np
import torch
from torch import nn


class SoftmaxClassifier(nn.Module):

    def __init__(self, encoder, conv_encoder, intermediate_fc, num_classes):

        super(SoftmaxClassifier, self).__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.conv_encoder = conv_encoder
        self.intermediate_fc = intermediate_fc

        self.create_fc_()
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.fc(self.encoder(x))
    
    def create_dummy(self, dummy_number=1):
        self.clf2 = nn.Linear(self.encoder.output_size, dummy_number, bias=True)
        nn.init.kaiming_normal_(self.clf2.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.kaiming_normal_(self.clf2.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.constant_(self.clf2.bias, 0)
    
    def pre2block(self, x):
        return self.conv_encoder(x)

    def dummypredict(self, x):
        return self.clf2(self.encoder(x))

    def latter2blockclf2(self, x):
        out = self.intermediate_fc(x)
        return self.clf2(out)

    def latter2blockclf1(self, x):
        out = self.intermediate_fc(x)
        return self.fc(out)

    def proser_prediction(self, x, temp=200):
        logits = self.forward(x)
        dummy_logits = self.dummypredict(x)
        maxdummylogit, _ = torch.max(dummy_logits, 1)
        maxdummylogit = maxdummylogit.view(-1, 1)
        totallogits = torch.cat((logits, maxdummylogit), dim=1)
        print(totallogits)
        embedding = nn.functional.softmax(totallogits/temp, dim=1)
        return torch.argmax(embedding)

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

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        acc_np = np.mean(np.equal(np.argmax(pred_np, axis=1).astype(np.int32), y_np.astype(np.int32)).astype(np.float32))

        return torch.mean(loss), acc_np

    def create_fc_(self):

        self.fc = nn.Linear(self.encoder.output_size, self.num_classes, bias=True)

        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.constant_(self.fc.bias, 0)
