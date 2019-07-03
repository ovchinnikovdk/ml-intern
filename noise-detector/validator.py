#!/usr/bin/python3
from sklearn.metrics import accuracy_score
import torch
import numpy as np
from torch.autograd import Variable
from datautils import batch_iterator
from model import NoiseDetector
import sys

def validate(model_path='trained/cnn.pth', data_path='data/val'):
    model = torch.load(model_path).cpu()
    predicted = np.array([])
    labels = np.array([])
    for x, y in batch_iterator(batch_size=200, data_path=data_path, shape=model.input_shape):
        x = Variable(torch.Tensor(x))
        x = np.array(list(map(lambda val: 1 if val >= 0.5 else 0, model(x).data.numpy())))
        predicted = np.concatenate((predicted, x))
        labels = np.concatenate((labels, y))
    return accuracy_score(labels, predicted)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(validate())
    else:
        print(validate(model_path=sys.argv[0], data_path=sys.argv[1]))
