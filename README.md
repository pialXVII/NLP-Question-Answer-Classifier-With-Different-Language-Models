# NLP Question-Answer Classifier With Different Language Models

Multi-class text classification project that categorizes question/answer pairs into topic
categories (e.g. *Science & Mathematics*, *Health*, *Sports*, *Education & Reference*,
*Entertainment & Music*, *Family & Relationships*) using classical machine learning and
several deep learning architectures, comparing two text representation methods: **TF-IDF**
and **Skip-gram (Word2Vec)** embeddings.

## Dataset

- **Source:** Question Answer Classification Dataset
- **Training set:** 93,333 samples
- **Test set:** 59,999 samples
- **Columns:** `QA Text` (raw HTML-tagged question/answer text), `Class` (topic label)

## Pipeline

1. Exploratory data analysis (class distribution, text length, word clouds)
2. Text preprocessing (HTML stripping, tokenization, stopword removal, lemmatization/stemming)
3. Label encoding
4. Feature extraction:
   - TF-IDF (unigrams + bigrams, max 10,000 features)
   - Skip-gram / Word2Vec (100-dimensional embeddings)
5. Model training and evaluation
6. Hyperparameter tuning
7. Result comparison and analysis

## Models Trained

**With TF-IDF**
- Logistic Regression
- Deep Neural Network (Dense layers)

**With Skip-gram embeddings**
- SimpleRNN, GRU, LSTM
- Bidirectional SimpleRNN, Bidirectional GRU, Bidirectional LSTM
- DNN (with global average pooling)

Each architecture was trained as a baseline and then hyperparameter-tuned, for **18 models total**.

## Results

| Rank | Model | Accuracy | F1-Score (Macro) |
|------|-------|----------|-------------------|
| 1 | GRU (Tuned) | 0.6718 | 0.6655 |
| 2 | Bidirectional GRU (Tuned) | 0.6667 | 0.6629 |
| 3 | GRU (Skip-gram) | 0.6642 | 0.6619 |
| 4 | Bidirectional GRU (Skip-gram) | 0.6630 | 0.6615 |
| 5 | DNN (Skip-gram, Tuned) | 0.6615 | 0.6614 |

**Best overall model:** GRU (Tuned) — outperforms the best classical ML model
(Logistic Regression, Tuned) by 1.34% accuracy and 0.0104 F1-score.

**Worst performing model:** SimpleRNN (Skip-gram) — 0.1782 accuracy, likely due to
vanishing gradients on long sequences without gating.

## Tech Stack

- Python, PyTorch (GPU/CUDA), TensorFlow/Keras
- scikit-learn, gensim, NLTK
- pandas, NumPy, matplotlib, seaborn, wordcloud


