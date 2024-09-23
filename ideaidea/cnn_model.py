# cnn_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class FeatureExtractionCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # CIFAR-10 as an example

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(-1, 64 * 8 * 8)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)

def train_cnn():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    model = FeatureExtractionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # Training for 10 epochs
        running_loss = 0.0
        I=0
        for images, labels in trainloader:
            I=I+1
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Turn: {I}")
        print(f"Epoch {epoch+1}, Loss: {running_loss}")

    # Save model
    torch.save(model.state_dict(), "cnn_feature_extractor.pth")

def load_cnn():
    model = FeatureExtractionCNN()
    model.load_state_dict(torch.load("cnn_feature_extractor.pth"))
    model.eval()
    #model.summary()
    return model

if __name__ == "__main__":
    train_cnn()
