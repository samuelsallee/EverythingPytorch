import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class relunet(nn.Module):
    def __init__(self):
        super(relunet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.ReLU()
        )
        self.convnet.to(device)

        tensor_for_shape = torch.randn((1, 28, 28), device=device)
        tensor_for_shape = self.forward_conv(tensor_for_shape)
        self.linear_shape = tensor_for_shape.shape[0] * tensor_for_shape.shape[1] * tensor_for_shape.shape[2]

        self.fc = nn.Linear(self.linear_shape, 10)
        self.fc.to(device)

    def forward_conv(self, x):
        return self.convnet(x)

    def forward(self, x):
        out = self.forward_conv(x)
        out = self.fc(out.view((-1, self.linear_shape)))
        return F.softmax(out)


class leakyrelunet(nn.Module):
    def __init__(self):
        super(leakyrelunet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.LeakyReLU(0.1)
        )
        self.convnet.to(device)

        tensor_for_shape = torch.randn((1, 28, 28), device=device)
        tensor_for_shape = self.forward_conv(tensor_for_shape)
        self.linear_shape = tensor_for_shape.shape[0] * tensor_for_shape.shape[1] * tensor_for_shape.shape[2]

        self.fc = nn.Linear(self.linear_shape, 10)
        self.fc.to(device)

    def forward_conv(self, x):
        return self.convnet(x)

    def forward(self, x):
        out = self.forward_conv(x)
        out = self.fc(out.view((-1, self.linear_shape)))
        return F.softmax(out)


data = torchvision.datasets.MNIST(download=True, transform=transforms.ToTensor(), root='', train=True)

dataloaders = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(download=True, transform=transforms.ToTensor(), root='', train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)


def train(model, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        for index, (img, label) in enumerate(dataloaders):
            img = img.to(device)

            label = label.to(device)

            model.zero_grad()

            output = model(img.view(-1, 1, 28, 28))
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()


def test(model):
    correct, total = 0, 0
    with torch.no_grad():
        for index, (img, label) in enumerate(testloader):
            img = img.to(device)
            label = label.to(device)
            output = model(img.view(1, 28, 28))
            if torch.argmax(output) == label:
                correct += 1
            total += 1
    print('Accuracy: ', correct / total)


model = relunet()
model2 = leakyrelunet()

model.to(device)
model2.to(device)

train(model)

test(model)
model2.load_state_dict(model.state_dict())
print(model2)
test(model2)