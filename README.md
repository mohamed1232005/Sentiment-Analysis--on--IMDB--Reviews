# 🎬 IMDB Sentiment Analysis System: NLP-based Movie Review Classifier

This repository contains a **complete end-to-end project** for performing **Sentiment Analysis on the IMDB Movie Reviews Dataset** using **Natural Language Processing (NLP)** and **Machine Learning**.  
The project includes **data exploration, preprocessing, model training, evaluation, error analysis, cross-validation, model comparison, hyperparameter tuning, interactive testing, and deployment using Flask + ngrok**.

---

## 📑 Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [1. Data Exploration (EDA)](#1-data-exploration-eda)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Train-Test Split](#3-train-test-split)
  - [4. Feature Engineering (TF-IDF)](#4-feature-engineering-tf-idf)
  - [5. Model Training](#5-model-training)
  - [6. Model Evaluation](#6-model-evaluation)
  - [7. Cross-Validation](#7-cross-validation)
  - [8. Model Comparison](#8-model-comparison)
  - [9. Hyperparameter Tuning](#9-hyperparameter-tuning)
  - [10. Error Analysis](#10-error-analysis)
  - [11. Interactive Testing](#11-interactive-testing)
  - [12. Deployment (Flask + ngrok)](#12-deployment-flask--ngrok)
- [Results](#results)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Future Work](#future-work)
- [License](#license)

---

## 📖 Introduction
**Sentiment Analysis** is a core task in NLP that involves determining whether a given piece of text expresses a **positive**, **negative**, or **neutral** opinion.  

Examples:
- `"I loved this movie!" → Positive`
- `"This movie was boring." → Negative`

Applications include:
- Customer feedback analysis  
- Social media monitoring  
- Market research  
- Recommendation systems  

In this project, we build a **binary classifier** (`positive` vs `negative`) for IMDB movie reviews.

---

## 🎬 Dataset
We use the **IMDB Movie Reviews Dataset** available on Kaggle:  
👉 [IMDB 50K Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### 📊 Dataset Details
| Attribute   | Description |
|-------------|-------------|
| **Reviews** | 50,000 movie reviews from IMDB |
| **Sentiment** | Labeled as either `positive` or `negative` |
| **Balance** | 25,000 positive reviews & 25,000 negative reviews (perfectly balanced) |
| **Type** | Text classification dataset |
| **Format** | CSV with two columns: `review` and `sentiment` |

### Example Rows
| review | sentiment |
|--------|-----------|
| "This movie was fantastic, I absolutely loved it!" | positive |
| "The plot was dull and boring, worst film ever." | negative |

---

## 🔄 Project Workflow

### 1. Data Exploration (EDA)
- Checked dataset shape, columns, datatypes, missing values, duplicates.
- Verified **class balance** → equal distribution of positive and negative reviews.
- Visualizations:
  - Class distribution barplot
  - Histogram of review lengths
  - Word clouds for positive & negative reviews

### 2. Preprocessing
Text preprocessing pipeline:
1. **Duplicate Removal**
2. **Lowercasing**
3. **HTML tag removal**
4. **Contraction expansion**  
   - e.g. `"can't"` → `"cannot"`, `"won't"` → `"will not"`
5. **Tokenization** using NLTK
6. **Stopword removal** (kept negations: `not, no, never`)
7. **Negation handling**  
   - e.g. `"not good"` → `"not_good"`
8. **Lemmatization** using WordNet

📌 Example:
Original: "I didn't like the movie; it wasn't good at all!"
After Preprocessing: ['not_like', 'movie', 'not_good']


### 3. Train-Test Split
- Performed an **80-20 split** (40K train, 10K test) with stratification to maintain balance.

### 4. Feature Engineering (TF-IDF)
- Used **TF-IDF Vectorizer** with:
  - `max_features=5000`
  - `ngram_range=(1,2)` (unigrams + bigrams)

### 5. Model Training
- **Logistic Regression** (baseline)  
- Compared later with **Naive Bayes** and **Support Vector Machine (SVM)**.

### 6. Model Evaluation
- Metrics:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC Score

Confusion Matrix Example:
  Predicted
          | Positive | Negative
--------------+----------+----------
Actual Positive | 4400 | 600
Actual Negative | 568 | 4432



### 7. Cross-Validation
- 5-fold Stratified Cross-Validation
- Results: accuracies stable around **87.8% – 88.4%**
- Mean accuracy ≈ **88.0%**

### 8. Model Comparison
| Model              | Accuracy |
|--------------------|----------|
| Naive Bayes        | 80.0%    |
| SVM (Linear Kernel)| 87.9%    |
| Logistic Regression| 88.2%    |

✅ Logistic Regression performed best, with SVM very close.

### 9. Hyperparameter Tuning
- Used **GridSearchCV** for Logistic Regression.
- Best parameters:
  - `C = 1`
  - `solver = liblinear`
- Best CV Score ≈ **87.9%**
- Final Test Accuracy ≈ **88.2%**

### 10. Error Analysis
- Misclassified reviews: **1168 / 10,000** (~11.6%)
- Errors often due to:
  - **Subtle or mixed sentiment**
  - **Sarcasm/irony**
  - **Very short/vague reviews**

### 11. Interactive Testing
Function to test custom reviews:
```python
def predict_sentiment(review):
    clean_tokens = preprocess_text(review)
    clean_text = " ".join(clean_tokens)
    vectorized = tfidf.transform([clean_text])
    prediction = best_lr.predict(vectorized)[0]
    proba = best_lr.predict_proba(vectorized)[0]
    return prediction, max(proba)
```

Example:
predict_sentiment("I absolutely loved this movie!")
Output → ("positive", 0.94)

### 12. Deployment (Flask + ngrok)

**Web App Features:**
- Built using Flask
- HTML form to enter reviews
- Real-time prediction display
- Confidence score shown with each prediction
- Color-coded outputs:
  - 🟢 Positive
  - 🔴 Negative
- Hosted via ngrok for public access

**Results:**
- **Best Model:** Logistic Regression + TF-IDF
- **Accuracy:** ~88%
- **ROC-AUC:** ~0.95
- **Cross-Validation:** Stable results, low variance
- **Error Rate:** ~11.6% misclassifications (mostly nuanced reviews)

**How to Run:**
- Run notebook:  
  ```bash
  jupyter notebook "Sentiment Analysis on IMDB Reviews.ipynb"
