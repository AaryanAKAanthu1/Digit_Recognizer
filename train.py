import numpy as np
import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt

torch.manual_seed(1725)
data = pd.read_csv("/data/train.csv")
y = data['label']
data = data.T
x = data[1:]
y = torch.tensor(y.T.values, dtype=torch.float32)
y = y.unsqueeze(-1)
x = torch.tensor(x.T.values, dtype=torch.float32)

def visualize(pixels):
    pixels = pd.DataFrame(pixels)
    image = pixels.values.reshape(28, 28)
    plt.imshow(image, cmap="gray")

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32,3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
           nn.Linear(32*26*26,256),
           nn.ReLU(),
           nn.Dropout(),
           nn.Linear(256,10)
       ) 

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        # print(x.shape)
        features = self.network(x)
        output = self.classifier(features)
        return output


model = ImageClassifier()
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
x = x.to(device)
y = y.to(device)

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(x, y)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

def train(steps):
    model.train()
    for i in range(steps):
        for imgs, lbls in loader:
            logits = model(imgs)
            lbls = lbls.view(-1)
            lbls = lbls.to(torch.long)
            loss = loss_fn(logits,lbls)
            y_pred = logits.argmax(dim=1)
            if torch.isnan(loss):
                print(loss)
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()
        if i%100 == 0:
            print(f"step:{i} Loss:{loss}")
          
def test(x,total_images):
    i = 0
    model.eval()
    predictions = torch.zeros(total_images)
    while i<total_images:
        logits = model(x[i:i+63])
        y_pred = logits.argmax(dim=1)
        predictions[i:i+63] = y_pred
        i += 64

    return predictions
  
if __name__ = '__main__':
  steps = 10000
  train(steps)
  y1 = y.squeeze()
  y_pred = test(x,42000)
  y_pred = y_pred.to(torch.int32)
  correct = (y_pred == y1.cpu()).sum()
  accuracy = 100*correct/len(y)
  print(f" Accuracy: {accuracy}")
  
