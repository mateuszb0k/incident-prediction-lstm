from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve,confusion_matrix,f1_score,average_precision_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
df = pd.read_csv('dataset.csv',header=None)
torch.manual_seed(42)
data = df.values
X = data[:,0]
y = data[:,1]
split_idx = int(0.8 * len(X))
X_train = X[:split_idx].reshape(-1,1)
y_train = y[:split_idx]
X_val = X[split_idx:].reshape(-1,1)
y_val = y[split_idx:]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
WINDOW_SIZE = 100
HORIZON_SIZE = 20
windowed_x_train = []
windowed_y_train = []
windowed_x_val = []
windowed_y_val = []
for i in range(0,len(X_train)-WINDOW_SIZE-HORIZON_SIZE):
    start = i
    end = min(i+WINDOW_SIZE, len(X_train))
    windowed_x_train.append(X_train[start:end])
    window_y = y_train[end:min(end+HORIZON_SIZE,len(X_train))]
    if 1 in window_y:
        windowed_y_train.append(1)
    else:
        windowed_y_train.append(0)
for i in range(0,len(X_val)-WINDOW_SIZE-HORIZON_SIZE):
    start = i
    end = min(i+WINDOW_SIZE, len(X_val))
    windowed_x_val.append(X_val[start:end])
    window_y = y_val[end:min(end+HORIZON_SIZE,len(X_val))]
    if 1 in window_y:
        windowed_y_val.append(1)
    else:
        windowed_y_val.append(0)
windowed_x_train = np.array(windowed_x_train)
windowed_y_train = np.array(windowed_y_train)
windowed_x_val = np.array(windowed_x_val)
windowed_y_val = np.array(windowed_y_val)
windowed_x_train = torch.tensor(windowed_x_train,dtype=torch.float32)
windowed_y_train = torch.tensor(windowed_y_train,dtype=torch.float32).unsqueeze(1)
windowed_x_val = torch.tensor(windowed_x_val,dtype=torch.float32)
windowed_y_val = torch.tensor(windowed_y_val,dtype=torch.float32).unsqueeze(1)
train_dataset = TensorDataset(windowed_x_train, windowed_y_train)
val_dataset = TensorDataset(windowed_x_val, windowed_y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
class LSTMModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim,hidden_dim,layer_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)
    def forward(self,x,h0=None,c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim,x.size(0),self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim,x.size(0),self.hidden_dim).to(x.device)
        out,(hn,cn) = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        # out = torch.sigmoid(out) no sigmoid with bcewithlogitsloss
        return out,hn,cn

model = LSTMModel(input_dim=1,hidden_dim=64,layer_dim=1,output_dim=1)
# criterion = nn.BCELoss()
num_zeros = (windowed_y_train==0).sum().item()
num_ones = (windowed_y_train==1).sum().item()
weight = num_zeros/(num_ones)
print(weight)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))#knowing that we have 100 arificial positives
optim = torch.optim.Adam(model.parameters())
EPOCHS = 30
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    train_loss = 0
    val_loss = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optim.zero_grad()
        out = model(data)[0]
        loss = criterion(out,target)
        loss.backward()
        train_loss += loss.item()
        optim.step()
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            out = model(data)[0]
            preds = (torch.sigmoid(out) >= 0.5).float()
            tp = ((preds==1)&(target==1)).sum().item()
            fp = ((preds==1)&(target==0)).sum().item()
            fn = ((preds==0)&(target==1)).sum().item()
            loss = criterion(out,target)
            val_loss += loss.item()
            total_tp += tp
            total_fp += fp
            total_fn += fn
    avg_val_loss = val_loss/len(val_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f'Best val loss at epoch {epoch+1} is {best_val_loss}')
    print(100*'-'+ f"EPOCH: {epoch+1}/{EPOCHS}")
    print(f'Train loss: {train_loss/len(train_loader):.4f} Validation loss: {val_loss/len(val_loader):.4f}')
    print(f'Recall {total_tp/(total_tp+total_fn+1e-7):.4f}')
    print(f'Precision {total_tp/(total_tp+total_fp+1e-7):.4f}')

test_model = LSTMModel(input_dim=1,hidden_dim=64,layer_dim=1,output_dim=1)
test_model.load_state_dict(torch.load('best_model.pt'))
test_model.eval()
out_arr = []
target_arr = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(val_loader):
        out = test_model(data)[0]
        out_arr.append(out)
        target_arr.append(target)
out_cat = torch.cat(out_arr,dim=0)
target_cat = torch.cat(target_arr,dim=0)
probs_flat = torch.sigmoid(out_cat).numpy().flatten()
targets_flat = target_cat.numpy().flatten()
precision, recall, thresholds = precision_recall_curve(targets_flat, probs_flat)
f1_scores = 2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-9)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
probs = probs_flat>=best_thresh
print(probs)
cm = confusion_matrix(targets_flat, probs)
f1_score = f1_score(targets_flat, probs)
average_precision = average_precision_score(targets_flat, probs_flat)
print(cm)
print(f1_score)
print(average_precision)