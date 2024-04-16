import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        # write your codes here
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Assuming 10 classes for image classification

    def forward(self, img):
        # write your codes here
        x = self.pool(torch.relu(self.conv1(img)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        # write your codes here
        super(CustomMLP, self).__init__()
        self.conv_layers = nn.Sequential(
            # C1 (5*5*1+1)*6 = 156
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                      stride=1, padding=2, bias=True),
            nn.Tanh(),
            # S2
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3 (5*5*6+1)*16 = 2416
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,
                      stride=1, padding=0, bias=True),
            nn.Tanh(),
            # S4
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            # C5 (16*5*5+1)*120 = 48120
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Tanh(),
            # F6 (120+1)*84 = 10164
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            # OUTPUT (84+1)*10 = 850
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )

        # Total number of parameters = 123,412
        # 156+2,416+48,120+10,164+850 = 61,706
        # backpropagation 61,706
    def forward(self, img):
        # write your codes here
        x = self.conv_layers(img)
        x = x.view(-1, 16 * 5 * 5)
        output = self.fc_layers(x)
        return output

class LeNet5moon(nn.Module):
    def __init__(self):
        super(LeNet5moon, self).__init__()  # 부모 클래스인 nn.Module의 생성자 호출
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, 10)  # 10개 클래스를 가정한 이미지 분류

    def forward(self, img):
        x = self.pool(F.relu(self.bn1(self.conv1(img))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(self.dropout1(x)))
        x = F.relu(self.fc2(self.dropout2(x)))
        output = self.fc3(x)
        return output
    
class LeNet5moon2(nn.Module):
    def __init__(self):
        super(LeNet5moon2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  # Adjust the dimensions according to your input size
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        x = self.pool(F.relu(self.bn1(self.conv1(img))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 5 * 5)  # Flatten the tensor before feeding into fully connected layer
        x = F.relu(self.fc1(self.dropout1(x)))
        x = F.relu(self.fc2(self.dropout2(x)))
        output = self.fc3(x)
        return output
    
