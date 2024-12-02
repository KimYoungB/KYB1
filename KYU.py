import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

dataraw = pd.read_csv(filepath_or_buffer:'data/BTC-USD.csv',
                        index_col ='Date',parse_dates = ['Date'])
dataset = pd.DataFrame(dataraw['Close'])

scaler = MinMaxScaler()
dataset_norm = dataset.copy()
dataset_norm['Close'] = scaler.fit_transform(dataset[['Close']])

totaldata = dataset.values
totaldatatrain = int(len(totaldata)*0.7)
totaldataval = int(len(totaldata)*0.1)
training_set = dataset_norm[0:totaldatatrain]
val_set = dataset_norm[totaldatatrain:totaldatatrain + totaldataval]
test_set = dataset_norm[totaldatatrain + totaldataval:]

def create_sliding_windows(data, len_data, lag):
    x, y=[], []
    for i in range(lag, len_data):
        x.append(data[i - lag:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

lag = 2
array_training_set = np.array(training_set)
array_val_set = np.array(val_set)
array_test_set = np.array(test_set)

x_train, y_train = create_sliding_windows(array_training_set, len(array_training_set), lag)
x_val, y_val = create_sliding_windows(array_val_set, len(array_val_set), lag)
x_test, y_test = create_sliding_windows(array_test_set, len(array_test_set), lag)

x_train, y_train = (torch.tensor(x_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32))
x_val, y_val = (torch.tensor(x_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32))
x_test, y_test = (torch.tensor(x_test, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.float32))

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel,self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=3,
                          batch_first=True,dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out

input_size = 1
hidden_size = 64
output_size = 1
model = GRUModel(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 1000
batch_size = 256

for epoch in range (epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train.unsqueeze(-1))
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if(epoch + 1) % 10 == 0:
        model.eval()
        val_outputs = model(x_val.unsqueeze(-1))
        val_loss = criterion(val_outputs.squeeze(), y_val)
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

model.eval()
y_pred = model(x_tes.unsqueeze(-1)).detach().numpy()
y_pred_invert_norm = scaler.inverse_transform(y_pred)

def rmse(datatest, datapred):
    return np.sqrt(np.mean((datapred - datatest)** 2))
def mape(datatest, datapred):
    return np.mean(np.abs((datatest - datapred) / datatest)) * 100

datatest = dataset['Close'][totaldatatrain + totaldataval + lag:].values
print('RMSE:', rmse(datatest, y_pred_invert_norm))
print('MAPE:', mape(datatest, y_pred_invert_norm))

plt.figure(figsize=(10, 4))
plt.plot(args:datatest, label='Data Test', color='red')
plt.plot(args:y_pred_invert_norm, label='Prediction Resluts', color='blue')
plt.title('Graph Comparison Data Actual and Data Prediction')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, output_size, num_layers = 3,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out