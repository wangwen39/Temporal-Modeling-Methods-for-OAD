import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class model(nn.Module):

    def __init__(self, input_size=4096, d_a=256, num_classes=31):
        super(model, self).__init__()

        self.linear_first = nn.Linear(input_size, d_a)

        self.linear_second = nn.Linear(d_a, 1)

        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(nn.Linear(input_size, num_classes), nn.Dropout(p=0.3))

    def forward(self, x):

        y = self.tanh(self.linear_first(x))

        y = self.linear_second(y)

        y = self.softmax(y)

        attention = y.transpose(1, 2)  # [16, 1, 35]

        embeddings = attention@x

        prob = self.classifier(embeddings.squeeze(1))

        return prob

