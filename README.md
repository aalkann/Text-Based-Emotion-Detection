# Emotion Classification Project

This repository contains a project focused on building and evaluating machine learning models for emotion classification using textual data. The dataset includes text samples labeled with various emotions, and the objective is to classify each sample into its respective emotion category.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Features and Models](#features-and-models)
4. [Results](#results)

---

## **Introduction**
This project applies Natural Language Processing (NLP) and machine learning techniques to classify emotions based on text. The project pipeline includes:
- Data preprocessing and text cleaning.
- Feature extraction using vectorization methods.
- Model training and evaluation using classification algorithms.

---

## **Dataset**
- **URL:** `https://www.kaggle.com/datasets/nelgiriyewithana/emotions`
- **Columns:**
  - `text`: Text data for emotion classification.
  - `label`: The emotion label for each text sample.
- **Preprocessing Steps:**
  - Sampling equal instances of each label for balanced training and testing.
  - Cleaning text by lowercasing and removing stopwords.

- **Class Labels:**
   1. Sadness = 0
   2. Joy = 1
   3. Love = 2
   4. Anger = 3
   5. Fear = 4
   6. Surprise = 5

---

## **Features and Models**

### **Feature Extraction**
1. **Count Vectorizer:** Converts text into a sparse matrix of token counts.
2. **TF-IDF Vectorizer:** Converts text into a matrix based on Term Frequency-Inverse Document Frequency scores.

### **Models**
1. **Multinomial Naive Bayes:**
   - Trained with Count Vectorizer and TF-IDF features.
2. **Logistic Regression:**
   - Uses TF-IDF features for classification.
3. **Support Vector Machine (SVM):**
   - Implements a linear kernel with TF-IDF features.

---

## **Results**
Model accuracy was evaluated on the test set using the following metrics:
- **Accuracy Score**
- **Classification Report**
- **Confusion Matrix Visualization**

| Model                                  | Accuracy | Prediction Time for 18 000 sample |
|----------------------------------------|----------|-----------------------------------|
| Naive Bayes with CountVectorizer       | 0.87      | 0.1s                              |
| Naive Bayes with TF-IDF                | 0.87      | 0.1s                              |
| Logistic Regression                    | 0.91      | 0.1s                              |
| Support Vector Machine (SVM)           | 0.91      | 30.9s                             |

A bar plot comparing model accuracy is included in the results.

---
