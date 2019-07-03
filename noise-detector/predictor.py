#!/usr/bin/python3
from sklearn.metrics import accuracy_score
import torch
import numpy as np
from torch.autograd import Variable
from datautils import reshape_mel
from model import NoiseDetector
import sys

def validate(model_path='trained/cnn.pth', file_path):
    model = torch.load(model_path).cpu()
    x = Variable(torch.Tensor(reshape_mel(np.load(file_path))[None]))
    val = model(x).data.numpy()[0]
    return 1 if val >= 0.5 else 0
    

if __name__ == '__main__':
    print(predict(file_path))
