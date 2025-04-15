import json
import gzip

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

import scipy.sparse as sp


def load(fn):
    with gzip.open(fn, "rt") as f:
        return json.load(f)


def extract(data):
    content = ["\n".join([a["title"]] + a["paragraphs"]) for a in data]
    topics = [[a.get("topics", "unknown")] for a in data]
    return content, topics


class RTVSlo:

    def fit(self, train_data: list):
        # here I used Ridge, which is sklearn's model,
        # but you MUST use pyTorch!
        self.model = Ridge(alpha=1)
        content, onehot = extract(train_data)
        self.vectorizer = TfidfVectorizer()
        X_text = self.vectorizer.fit_transform(content)
        self.onehot = OneHotEncoder(handle_unknown="ignore", max_categories=50)
        oh = self.onehot.fit_transform(onehot)
        X = sp.hstack([X_text, oh])
        y = np.array([d['n_comments'] for d in train_data])
        self.model.fit(X, y)

    def predict(self, test_data: list):
        content, onehot = extract(test_data)
        X_text = self.vectorizer.transform(content)
        oh = self.onehot.transform(onehot)
        X = sp.hstack([X_text, oh])
        return self.model.predict(X)


if __name__ == '__main__':

    # this shows how your solution should be called

    train = load("data/rtvslo_train.json.gz")
    test = load("data/rtvslo_test.json.gz")

    m = RTVSlo()
    m.fit(train)

    p = m.predict(test)

    np.savetxt('example.txt', p, fmt='%f')
