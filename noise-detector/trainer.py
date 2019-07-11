#!/usr/bin/python3
import torch
from torch.autograd import Variable
from torch.optim import Adam
from datautils import batch_iterator
from model import NoiseDetector
import sys
import os
from tqdm import tqdm
import time

def train(net, path, batch_size=100, n_epochs=30, lr=1e-4):
    optimizer = Adam(net.parameters(), lr=lr)
    loss = torch.nn.BCELoss()
    for i in tqdm(range(n_epochs), desc='Training epochs'):
        sum_loss = 0
        for x, y in batch_iterator(batch_size=batch_size, data_path=path, shape=net.input_shape):
            x = Variable(torch.Tensor(x)).cuda()
            y = Variable(torch.Tensor(y)).cuda()
            optimizer.zero_grad()
            output = net(x)
            loss_out = loss(output, y)
            loss_out.backward()
            optimizer.step()
            sum_loss += loss_out.data[0]
        if i % 5 == 0 and i > 0:
            print("EPOCH #" + str(i))
            print("Loss: " + str(sum_loss))
            if i % 10 == 0:
                torch.save(net, 'trained/cnn_epoch' + str(i) + '.pth')
    torch.save(net, 'trained/cnn.pth')


if __name__ == '__main__':
    model = NoiseDetector().cuda()
    if len(sys.argv) != 4:
        train(model, os.path.join('data', 'train'), batch_size=256, n_epochs=10, lr=1e-4)
    else:
        train(model, sys.argv[0], int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
