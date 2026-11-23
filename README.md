# PaySim Temporal Fraud Detection

## Project Overview

This project focuses on detecting fraudulent financial transactions using temporal deep learning architectures on the PaySim synthetic financial dataset.

The main objective is to investigate how sequential transaction patterns can be used for fraud detection by comparing a baseline LSTM model with a hybrid LSTM and Temporal Attention architecture.

## Dataset

- **Dataset:** PaySim
- **Source:** [Kaggle - PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Total Transactions:** 6,362,620
- **Fraudulent Transactions:** 8,213
- **Time Feature:** `step`
- **Target Variable:** `isFraud`
- **Number of Time Steps:** 743

The dataset contains highly imbalanced transaction records, making fraud detection a challenging binary classification task.

## Data Preprocessing

The preprocessing workflow includes:

- Sorting transactions chronologically using the `step` column.
- Removing identifier columns (`nameOrig` and `nameDest`).
- Excluding the `isFlaggedFraud` feature.
- One-hot encoding transaction types.
- Applying StandardScaler to numerical and encoded features.
- Splitting the dataset chronologically into training (70%), validation (15%), and testing (15%) sets.
- Constructing sequences using a sliding window of five consecutive transaction records.
- Handling class imbalance using weighted binary cross-entropy loss.

## Model Architectures

Two deep learning architectures were implemented:

### 1. Baseline LSTM

- Two-layer LSTM network.
- Hidden size: 64.
- Dropout: 0.2.
- Fully connected output layer.
- Binary fraud classification.

### 2. Hybrid LSTM + Temporal Attention

This architecture combines a two-layer LSTM network with a temporal attention mechanism.

The attention layer assigns weights to different timesteps in the input sequence and generates a weighted context representation for fraud classification.

The architecture also includes fully connected layers, ReLU activation, and dropout.

## Model Evaluation

The models were evaluated using PR-AUC and ROC-AUC on the test set.

| Evaluation Metric | Baseline LSTM | LSTM + Attention |
|---|---:|---:|
| PR-AUC | 0.4882 | 0.4700 |
| ROC-AUC | 0.8402 | 0.8368 |
| Fraud Recall | 0.6382 | 0.6465 |

## Temporal Attention Analysis

The hybrid model includes an attention-weight analysis to investigate how different timesteps contribute to its predictions.

The average attention weights were calculated separately for fraudulent and non-fraudulent transactions.

This analysis provides an additional perspective on the temporal patterns captured by the model.

## Technologies & Libraries

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- KaggleHub

## Repository Structure

```text
paysim-temporal-fraud-detection/
│
├── README.md
└── paysim.Temporal.ipynb
```

## How to Run

1. Clone or download this repository.
2. Open the notebook in Google Colab or Jupyter Notebook.
3. Install the required Python libraries.
4. Ensure access to the PaySim dataset through KaggleHub.
5. Enable GPU acceleration for faster model training.
6. Run the notebook cells sequentially.

## Project Scope

This project is an educational implementation exploring temporal deep learning approaches for financial fraud detection, with a focus on LSTM networks, Temporal Attention, and attention-weight analysis.

---

**Author:** Ayda Pirandakh
