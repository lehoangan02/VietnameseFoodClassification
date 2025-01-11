import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
from torchvision import transforms

import food_dataset as fd

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
class VNFNeuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100*100*3, 10000),  # Adjust input size to 100x100
            nn.ReLU(),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 2500),
            nn.ReLU(),
            nn.Linear(2500, 512),
            nn.ReLU(),
            nn.Linear(512, 30)  # Adjust output size to 30 labels
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# model = VNFNeuNet().to(device)
# model = models.vgg16(weights='IMAGENET1K_V1')
# model.classifier[6] = nn.Linear(4096, 10)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    TrainDataset = fd.VietnameseFoodDataset("TrainLabels.csv", "./Train")

    TrainDataLoader = DataLoader(TrainDataset, batch_size=8, shuffle=True, num_workers=4)

    TestDataLoader = DataLoader(TrainDataset, batch_size=8, num_workers=6)
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier[6] = nn.Linear(4096, 30)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    # epoch = 1

    # for t in range(epoch):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train(TrainDataLoader, model, loss_fn, optimizer)
    #     test(TestDataLoader, model, loss_fn)
    # model = models.vgg16(weights='IMAGENET1K_V1')
    # model.classifier[6] = nn.Linear(4096, 30)
    model = torch.load('model.pth', weights_only=False)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    test(TestDataLoader, model, loss_fn)
    print("Done!")
    # torch.save(model, 'model.pth')