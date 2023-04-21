import itertools
import os

import torch_directml
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import pickle

dml = torch_directml.device()


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float, device=dml)
        self.y = torch.tensor(y, dtype=torch.float, device=dml)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


class Model(nn.Module):

    def __init__(self, input_size=22, hidden_size_1=35, hidden_size_2=18, output_size=6):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def save_weights(self):
        torch.save(self.state_dict(), f'..{os.sep}data{os.sep}model_weights.pt')

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float, device=dml)
        return self(X)

    def load_weights(self):
        self.load_state_dict(torch.load(f'..{os.sep}data{os.sep}model_weights.pt'))


def get_tensor_data():
    with open(f'..{os.sep}data{os.sep}tensor_ready_data.pkl', 'rb') as f:
        tensor_ready_data = pickle.load(f)

    inputs, outputs = tensor_ready_data

    # Scale the data to a range of 0 to 1 using sklean min max scaler
    scaler = MinMaxScaler()
    inputs = scaler.fit_transform(inputs)
    outputs = scaler.fit_transform(outputs)

    test_in, test_out = [], []
    inputs = inputs.tolist()
    outputs = outputs.tolist()
    # Remove 20 % of the data for testing
    twenty_percent = int(len(inputs) * 0.2)
    for _ in range(twenty_percent):
        rand = torch.randint(0, len(inputs), (1,)).item()
        test_in.append(inputs.pop(rand))
        test_out.append(outputs.pop(rand))

    batch_size = 96
    train_data = Data(inputs, outputs)
    test_data = Data(test_in, test_out)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model, train_loader, test_loader):
    model.to(dml)
    # make sure model is on GPU dml
    print(next(model.parameters()).device)
    learning_rate = 0.1
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)
    num_epochs = 100000
    for epoch in range(num_epochs):
        for X, y in train_loader:
            optimizer.zero_grad()

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            model.save_weights()

        scheduler.step()


def test(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in test_loader:
            y_pred = model(X)
            # If both predictions and actual values are greater than 0.5, then it is a correct prediction, or if both
            # are less than 0.5, then it is a correct prediction
            mood_pred = y_pred.tolist()
            mood_actual = y.tolist()
            while mood_pred:
                predicted = mood_pred.pop()
                actual = mood_actual.pop()
                # Find the index of the max value in the list
                predicted = predicted.index(max(predicted))
                actual = actual.index(max(actual))
                if predicted == actual:
                    correct += 1
                total += 1

        print(f'Accuracy of the model on the test data: {correct / total * 100} %')
