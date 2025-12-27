## Comment Prediction for RTVSlo News

This project focuses on predicting the number of comments that will be written on news articles scraped from [rtvslo](https://www.rtvslo.si/). The goal is to build models that, given the text and metadata of a news article, estimate how many comments it will receive.

### Data

Each article includes:
- Title
- Paragraphs
- Leads
- Author
- Publication date and time
- Topic and subtopic labels

The dataset is split into training and test sets, and temporal information is used extensively to improve predictions.

### Objective

Build and evaluate models that predict the number of comments an article will receive. The primary evaluation metric is Mean Absolute Error (MAE).

Predictions may be transformed (e.g., log scale) to better handle heavy-tailed distributions of comment counts.

### Modeling Approaches

Various modeling strategies were explored:

#### Embeddings
- Text embeddings using TF-IDF and SloBerta
- Temporal embeddings (scaled year + cyclic weekday/hour)
- Topic and subtopic embeddings (learned vs one-hot)

#### Predictive Models
- Multilayer Perceptrons (MLP)
- LSTM
- Gradient Boosting
- Linear Regression
- KNN

#### Prediction Targets
- Raw count of comments
- Transformed targets:
  - $\ln(1 + \text{n\_comments})$
  - $\sqrt{\text{n\_comments}}$
  - Zero-Inflated Negative Binomial (ZINB)

### Model Selection and Evaluation

- Models were trained with early stopping based on validation MAE.
- A validation split of 95% training / 5% validation was used.
- Final evaluation was performed on a held-out test set.
- Models were compared based on MAE and the distribution of predictions.

### Final Model

The selected final model is an ensemble of three identical MLP networks trained on combined embeddings (text, topics, temporal features). Despite using the same seed, multiple training runs are averaged to mitigate variability in results.

<img src="https://mpog.dev/content/uozp/comments/model.png" style="width: 100%; max-width: 300px;" />

### Hyperparameter Tuning

Hyperparameters (e.g., learning rate, batch size, dropout, weight decay) were chosen using grid search and evaluated via cross-validation. Final hyperparameters include:
- Loss: L1 / Huber
- Batch size: 150
- Learning rate: 1e-4
- Weight decay: 1e-3
- Dropout: 0.1
- Hidden dimension: 1024

### Additional Analyses

The study also explored:
- Distribution-aware training
- Postprocessing strategies (quantile smoothing)
- Additional feature engineering (contextual and semantic features)
- Cluster-specific modeling

## Repository Structure
- ```models/``` directory containing pretrained models for the ensemble
- ```data/``` directory for datasets
- ```embeddings/``` directory containing SloBERTa embeddings of the data
- ```predstavitev/``` directory with presentation-related files
- ```workspace/``` directory with development environment files (not part of the final submission)
- ```final.py``` script for running the model that predicts the number of comments
- ```predstavitev.pdf``` PDF file with the presentation
- ```requirements.txt``` file listing the required libraries

## Running the Visualization and Setting Up the Environment
The visualization is intended to be run with `python 3.12`. Required libraries can be installed using:

```bash
pip install -r requirements.txt
```

Running the `final.py` script without additional arguments performs prediction on the test dataset `data/rtvslo_test.json` using the SloBERTa embeddings stored in `embeddings/sloberta_embeddings_test.pt`.
With additional arguments, it is possible to specify a custom test dataset and embeddings, generate SloBERTa embeddings, or train a new ensemble model.

### Arguments
- `--train`: train the model `model_01.pt` and save it to the `models/` directory
- `--embed`: enable embedding mode
- `--data_path`: path to the JSON file containing the data
- `--emb_path`: path to the SloBERTa embeddings file

### Examples
```bash
python final.py
python final.py --train
python final.py --embed --data_path test_data.json --emb_path embeddings/test_data.pt
python final.py --data_path test_data.json --emb_path embeddings/test_data.pt
```


python final.py --data_path test_data.json --emb_path embeddings/test_data.pt
```
