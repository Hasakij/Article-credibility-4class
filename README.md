# Article Credibility 4-Class Classifier

Multi-class NLP classifier for article credibility assessment.

## Classes

- True
- Fake
- Satire
- Bias

## Model Architecture

The classifier consists of:

- Embedding layer
- Conv1D layer
- ReLU activation
- MaxPooling
- BiLSTM
- Dropout
- Fully Connected output layer

## Dataset

Dataset source:
```text
https://www.kaggle.com/datasets/aviseth20/multi-class-fake-news-dataset
```

Expected dataset location:

```text
dataset/Dataset_Clean.csv
```


Class mapping:

```text
0 = TRUE
1 = FAKE
2 = SATIRE
3 = BIAS
```

The preprocessing pipeline combines article title and content into a single input text.

## Installation

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run training:

```bash
python3 train.py
```

The script:

- Loads and preprocesses the dataset
- Tokenizes text using XLM-RoBERTa tokenizer
- Splits data into train, validation and test sets
- Trains a CNN + BiLSTM classifier
- Saves the best checkpoint
- Logs metrics to Comet ML

## Example .env:

```text
COMET_API_KEY=your_api_key
```

## Prediction

Classify raw text:

```bash
python3 predict.py --text "Example article text"
```

Classify URL:

```bash
python3 predict.py --url "https://example.com/article"
```

Classify PDF:

```bash
python3 predict.py --pdf article.pdf
```

Example output:

```text
Prediction: SATIRE
REAL probability: 0.08
FAKE probability: 0.02
SATIRE probability: 0.90
BIAS probability: 0.00
```

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

## Example performance
Final evaluation on the test set:

```text
Accuracy: 0.81
Macro Precision: 0.82
Macro Recall: 0.85
Macro F1-score: 0.81
```

Class-wise F1-scores:

```text
TRUE   : 0.86
FAKE   : 0.83
SATIRE : 0.98
BIAS   : 0.58
```
The model performs very well on SATIRE class, while the BIAS class is sometimes confused with TRUE and FAKE classes
## Requirements

Main libraries:

- torch
- transformers
- pandas
- numpy
- scikit-learn
- spacy
- trafilatura
- pdfplumber
- comet_ml

## Git Ignore

The repository ignores:

```text
.env
best_model.pt
dataset/
__pycache__/
*.pyc
test.py
```
