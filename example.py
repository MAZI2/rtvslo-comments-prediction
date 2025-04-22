import json
import gzip

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

import scipy.sparse as sp

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def load(fn):
    with gzip.open(fn, "rt") as f:
        return json.load(f)


def extract(data):
    content = ["\n".join([a["title"]] + a["paragraphs"]) for a in data]
    topics = [[a.get("topics", "unknown")] for a in data]
    return content, topics


device = "cpu"


def dense_to_dataset(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1).float())


def torch_fit(dataset, lambda_=0, batch_size=1000, lr=0.1, epochs=30, collate_fn=None):
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

    class Linear(nn.Module):
        def __init__(self, inputs):
            super().__init__()
            self.linear = nn.Linear(inputs, 1)

        def forward(self, x):
            return self.linear(x)

    model = Linear(dataset[0][0].shape[0])
    model = model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_)

    def train(dataloader, model, loss_fn, optimizer):
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
        test_loss /= num_batches
        correct /= size
        print(f"Avg loss: {test_loss:>8f} \n")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(loader, model, loss_fn, optimizer)
        if (t+1) % 5 == 0:
            test(loader, model, loss_fn)

    return model

def torch_predict(model, dataset):
    model.eval()
    x = dataset[:][0]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
    return np.array(pred).reshape(-1)


class RTVSlo:

    def fit(self, train_data: list):
        content, onehot = extract(train_data)
        self.vectorizer = TfidfVectorizer(min_df=10, max_features=20000)  # avoid needing sparse data
        X_text = self.vectorizer.fit_transform(content)
        self.onehot = OneHotEncoder(handle_unknown="ignore", max_categories=50)
        oh = self.onehot.fit_transform(onehot)
        X = sp.hstack([X_text, oh]).astype(np.float32)
        X = X.toarray()
        y = np.array([d['n_comments'] for d in train_data])
        self.torch_model = torch_fit(dense_to_dataset(X, y), lambda_=0.00001)

    def predict(self, test_data: list):
        content, onehot = extract(test_data)
        X_text = self.vectorizer.transform(content)
        oh = self.onehot.transform(onehot)
        X = sp.hstack([X_text, oh]).astype(np.float32)
        X = X.toarray()
        return torch_predict(self.torch_model,
                             dense_to_dataset(X, np.zeros(len(X))))


if __name__ == '__main__':

    # this shows how your solution should be called

    train = load("data/rtvslo_train.json.gz")
    test = load("data/rtvslo_test.json.gz")

    m = RTVSlo()
    m.fit(train)

    p = m.predict(test)

    np.savetxt('example.txt', p, fmt='%f')
