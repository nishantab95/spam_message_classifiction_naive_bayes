📩 SMS Spam Message Classification using Naive Bayes

This project implements an end-to-end Natural Language Processing (NLP) pipeline to classify SMS messages as spam or ham (non-spam) using Naive Bayes classifiers and Bag-of-Words text features.

📌 Project Overview

Spam detection is a classic text classification problem in machine learning. In this project, we build and evaluate Naive Bayes models to automatically identify spam messages from SMS text data.

The workflow covers:

Data loading from TSV format

Text preprocessing

Feature extraction using Bag-of-Words

Model training with Multinomial Naive Bayes and Bernoulli Naive Bayes

Model evaluation and comparison

🚀 Features

✔ Load dataset from Kaggle or local environment (robust file handling)

✔ Text vectorization using CountVectorizer (Bag-of-Words)

✔ Stopword removal

✔ Multinomial Naive Bayes model (final selected model)

✔ Bernoulli Naive Bayes model (baseline comparison)

✔ Model evaluation using Accuracy, Precision, Recall, F1-score

✔ Confusion matrix analysis

🧠 Models Used

Multinomial Naive Bayes
Best suited for text data where word frequency matters.
✔ Final model selected due to better balanced performance.

Bernoulli Naive Bayes
Uses binary word presence features.
✔ Used for comparison and baseline evaluation.

📊 Results

Multinomial Naive Bayes

Accuracy: 98.9%

Strong precision and recall for both spam and non-spam classes

Very low false positive and false negative rates

Bernoulli Naive Bayes

Accuracy: 97.4%

High spam recall but higher false positives on non-spam messages

Conclusion:
Multinomial Naive Bayes provided more balanced and reliable performance and was selected as the final model.

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Jupyter Notebook

📁 Project Structure
spam_message_classification_naive_bayes/
│
├── spam_classifier.ipynb        # Main notebook
├── spam.tsv                    # Dataset (or Kaggle input)
├── README.md                   # Project documentation
└── requirements.txt            # (Optional) dependencies

▶️ How to Run

Clone the repository

Install dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook spam_classifier.ipynb


Run all cells to train and evaluate the model.

🔮 Future Improvements

Try TF-IDF instead of Bag-of-Words

Compare with Logistic Regression and SVM

Perform hyperparameter tuning

Build a small Streamlit web app for live spam prediction

📚 Dataset

SMS Spam Collection Dataset (TSV format)

Commonly used benchmark dataset for text classification tasks

🙌 Acknowledgements

UCI / Kaggle SMS Spam dataset

Scikit-learn documentation
