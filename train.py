from dataset import FaceLandmarksDatasetWithMediapipe
from model import LierDetectModel_v1 as LierDetectModel
from model import LierDetectModelWithCNN as CNN_MODEL
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import torchsummary
# from torchviz import make_dot


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        landmark, heart_rate = X

        pred = model(landmark, heart_rate)
        loss = loss_fn(pred, y)
        
        # BackPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for (landmark, heart_rate), y in dataloader:
            pred = model(landmark, heart_rate)
            test_loss += loss_fn(pred, y).item()
            correct += (pred >= torch.FloatTensor([0.5])).float().sum().item()

    test_loss /= num_batches
    correct /= size * num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def main_v1():
    dataset = FaceLandmarksDatasetWithMediapipe(csv_file="./data.csv")

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    # validation_size = dataset_size - train_size - validation_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)


    n_epochs = 20
    learning_rate = 0.01

    model = LierDetectModel()

    # _x = torch.zeros(1, 2, 10)
    # torchsummary.summary(model, ((3, 784, 1), (10, 1)))
    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    print("Done!")

    PATH = "./weights"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(model, os.path.join(PATH, "model_v1.pt"))

    #     torch.save({
    #     'model': model.state_dict(),
    #     'optimizer': optimizer.state_dict()
    # }, PATH + 'all.tar')


def train_loop_v2(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        landmark, heart_rate = X

        landmark =landmark.reshape(landmark.shape[0], 3, -1)

        pred = model(landmark, heart_rate)
        loss = loss_fn(pred, y)
        
        # BackPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop_v2(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for (landmark, heart_rate), y in dataloader:
            landmark =landmark.reshape(landmark.shape[0], 3, -1)
            pred = model(landmark, heart_rate)
            test_loss += loss_fn(pred, y).item()
            correct += (pred >= torch.FloatTensor([0.5])).float().sum().item()

    test_loss /= num_batches
    correct /= size * num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def main_v2():
    dataset = FaceLandmarksDatasetWithMediapipe(csv_file="./data.csv")

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    # validation_size = dataset_size - train_size - validation_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)


    n_epochs = 20
    learning_rate = 0.01

    model = CNN_MODEL()

    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(0, n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop_v2(train_dataloader, model, loss_fn, optimizer)
        test_loop_v2(test_dataloader, model, loss_fn)

    print("Done!")

    PATH = "./weights"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(model, os.path.join(PATH, "model_v2.pt"))


if __name__ == "__main__":
    main_v2()