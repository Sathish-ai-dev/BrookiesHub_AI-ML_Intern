# 🐦 Tweet Sentiment Classification (Positive/Negative)

This project applies **Natural Language Processing (NLP)** and **Machine Learning** to classify tweets as **positive** or **negative**.  

---

## 🎯 Objective
- Preprocess raw tweet text.  
- Vectorize text using **TF-IDF / CountVectorizer**.  
- Train ML models (e.g., Logistic Regression).  
- Evaluate performance with accuracy, confusion matrix, and classification report.  

---

## 📂 Dataset
- A CSV file (`tweets.csv`) with at least:  
  - `tweet`: text of the tweet  
  - `label`: sentiment (1 = positive, 0 = negative)  

You can use [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) or any Twitter dataset.

---

## 🛠️ Tools & Libraries
- **Python 3.x**  
- [Pandas](https://pandas.pydata.org/) – data manipulation  
- [NumPy](https://numpy.org/) – numerical operations  
- [NLTK](https://www.nltk.org/) – text preprocessing, stopwords  
- [scikit-learn](https://scikit-learn.org/) – ML models & evaluation  
- [Seaborn](https://seaborn.pydata.org/) – visualizations  

Install dependencies:
```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib
