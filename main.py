import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MNIST  # Assuming you have implemented the MNIST dataset
from model import LeNet5, CustomMLP, LeNet5moon, LeNet5moon2  # Assuming you have implemented the model

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = len(trn_loader.dataset)

    if total_samples == 0:
        print("No Train Dataset")
        return 0.0, 0.0  # 빈 데이터셋이면 0.0을 반환

    for batch_idx, (images, labels) in enumerate(trn_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

    trn_loss = total_loss / len(trn_loader)
    acc = 100.0 * correct / total_samples
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = len(tst_loader.dataset)  # 테스트 데이터셋의 총 샘플 수

    if total_samples == 0:
        print("No Test Dataset")
        return 0.0, 0.0  # 빈 데이터셋이면 0.0을 반환

    with torch.no_grad():
        for images, labels in tst_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

    tst_loss = total_loss / len(tst_loader)
    acc = 100.0 * correct / total_samples
    return tst_loss, acc

def modul_train_and_test(model, device, train_dataset, test_dataset, batch_size, train_loop):
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    results = []
    # Training loop
    for epoch in range(train_loop):
        trn_loss, trn_acc = train(model, train_loader, device, criterion, optimizer)
        tst_loss, tst_acc = test(model, test_loader, device, criterion)
        results.append({epoch: epoch, trn_loss: trn_loss, trn_acc: trn_acc, tst_loss: tst_loss, tst_acc: tst_acc})
        print(f"Epoch [{epoch+1}/10]\tTrain Loss: {trn_loss:.4f}\tTrain Acc: {trn_acc:.2f}%\tTest Loss: {tst_loss:.4f}\tTest Acc: {tst_acc:.2f}%")
    return

def main():
    train_tar_file = "../data/train.tar"
    test_tar_file = "../data/test.tar"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate your dataset and data loaders
    train_dataset = MNIST(train_tar_file)
    test_dataset = MNIST(test_tar_file)
      
    lenet5_results = modul_train_and_test(LeNet5(), device, train_dataset, test_dataset, 64, 10)
    custom_results = modul_train_and_test(CustomMLP(), device, train_dataset, test_dataset, 64, 10)
    moon_results = modul_train_and_test(LeNet5moon(), device, train_dataset, test_dataset, 64, 10)
    moon2_results = modul_train_and_test(LeNet5moon2(), device, train_dataset, test_dataset, 64, 10)

if __name__ == '__main__':
    main()
