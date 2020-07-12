# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, input_size=4096, hidden_size=4096, num_layers=1, num_classes=31):
        super(model, self).__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.num_classes = num_classes

        # self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)  # input:[N, C, L]

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)

        self.classifier = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Dropout(p=0.3))

    def forward(self, input_tensor):

        h0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size).cuda()

        output_tensor, hidden_state = self.lstm(input_tensor, (h0, c0)) 
        # avg_out = self.avgpool(output_tensor.transpose(2, 1))
        # prob = self.classifier(avg_out.squeeze(2))

        prob = self.classifier(output_tensor[:, -1, :])

        return prob

if __name__ == "__main__":

    input_tensor = torch.randn(64, 8, 4096).cuda()

    model = model(input_size=4096, num_classes=31).cuda()
    output = model(input_tensor)
