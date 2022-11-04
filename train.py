import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import torchsummary
from torch.utils.tensorboard import SummaryWriter
# from torchviz import make_dot

from dataset import FaceLandmarksDatasetWithMediapipe
from model import LierDetectModel_v2 as LierDetectModel
from model import LierDetectModelWithCNN as CNN_MODEL
from utils import create_directory


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        landmark, heart_rate = X

        landmark =landmark.reshape(landmark.shape[0], 3, -1)

        pred = model(landmark, heart_rate)
        loss = loss_fn(pred, y)
        
        # BackPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 100 == 0:
            _loss, current = loss.item(), batch * len(X)
            print(f"loss: {_loss:>7f} [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    return train_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()

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
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct 


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


    n_epochs = 50
    learning_rate = 0.01

    model = LierDetectModel()

    # TensorBoard Start
    writer = SummaryWriter()

    x1 = torch.zeros(1, 478, 3)
    x1 = x1.reshape(x1.shape[0], 3, -1)
    x2  = torch.zeros(1, 10)

    writer.add_graph(model, [x1, x2])

    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs):
        running_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        accuracy = test_loop(test_dataloader, model, loss_fn)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss:.5f}, Accuracy: {accuracy:.5f}")

        writer.add_scalar('training loss', running_loss, epoch+1)
        writer.add_scalar('test accuracy', accuracy, epoch+1)


    print("Done!")

    PATH = ["./weights"]
    create_directory(PATH)

    torch.save(model, os.path.join(PATH[0], "model_v1-4.pt"))

    # writer.close()



def train_loop_v2(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        landmark, heart_rate = X

        landmark =landmark.reshape(landmark.shape[0], 3, -1)

        pred = model(landmark, heart_rate)
        loss = loss_fn(pred, y)
        
        # BackPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # if batch % 100 == 0:
        #     _loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {_loss:>7f} [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    return train_loss


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


    return correct



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
    learning_rate = 0.1

    model = CNN_MODEL()

    writer = SummaryWriter()

    x1 = torch.zeros(1, 478, 3)
    x1 = x1.reshape(x1.shape[0], 3, -1)
    x2  = torch.zeros(1, 10)

    writer.add_graph(model, [x1, x2])

    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(0, n_epochs):
        running_loss = train_loop_v2(train_dataloader, model, loss_fn, optimizer)
        accuracy = test_loop_v2(test_dataloader, model, loss_fn)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss:.5f}, Accuracy: {accuracy:.5f}")

        writer.add_scalar('training loss', running_loss, epoch+1)
        writer.add_scalar('test accuracy', accuracy, epoch+1)

    print("Done!")

    PATH = "./weights"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(model, os.path.join(PATH, "model_v2.pt"))


if __name__ == "__main__":
    main_v1()