# 🎬 Movie Recommendation System (Collaborative Filtering)

This project builds a **collaborative filtering-based recommendation system** using the **MovieLens dataset** from Kaggle.  
It recommends movies to users based on **user similarity** and their past ratings.

---

## 🎯 Objective
- Build a **User-Based Collaborative Filtering** recommender.  
- Use **cosine similarity** on the user-item matrix.  
- Provide movie recommendations for a given user.  

---

## 📂 Dataset
We use the **MovieLens dataset** from Kaggle:  
👉 [MovieLens Dataset (100k)](https://www.kaggle.com/datasets/grouplens/movielens-100k)  

- `ratings.csv`: userId, movieId, rating, timestamp  
- `movies.csv`: movieId, title, genres  

---

## 🛠️ Tools & Libraries
- Python 3.x  
- [Pandas](https://pandas.pydata.org/) – data handling  
- [NumPy](https://numpy.org/) – numerical operations  
- [Scikit-learn](https://scikit-learn.org/) – cosine similarity  
- [Matplotlib / Seaborn](https://matplotlib.org/) – visualizations  

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
