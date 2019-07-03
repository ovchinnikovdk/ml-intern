import torch


class NoiseDetector(torch.nn.Module):
    """Noise Detector Convolutional Neural network"""
    def __init__(self):
        super(NoiseDetector, self).__init__()
        input_shape = (100, 80)
        self.input_shape = input_shape
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1)
        self.conv2 = torch.nn.Conv2d(10, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1)
        self.conv3 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1)
        self.linsize = (self.input_shape[0] - 3) * (self.input_shape[1] - 3)
        self.fc1 = torch.nn.Linear(8 * self.linsize, 30)
        self.dropout = torch.nn.Dropout(0.6)
        self.fc2 = torch.nn.Linear(30, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, input):
        out = self.conv1(input)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = out.view(-1, 8 * self.linsize)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.act(out)
        return out
