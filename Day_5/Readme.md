#  Customer Segmentation using K-Means Clustering

This project applies **K-Means clustering** to segment customers based on their purchasing behavior.  
Customer segmentation is a key part of marketing and helps businesses target different groups effectively.

---

##  Objective
- Cluster customers into distinct groups based on their spending patterns.  
- Use the **Elbow Method** to find the optimal number of clusters.  
- Visualize customer clusters with scatter plots.  

---

##  Dataset
We use a sample dataset such as **Mall Customers** dataset:  
- Features: `Annual Income (k$)`, `Spending Score (1-100)`  
- Each row represents a customer’s purchasing profile.  

Alternatively, you can replace this dataset with your own customer transaction data.

---

##  Tools & Libraries
- Python 3.x  
- [Pandas](https://pandas.pydata.org/) – data handling  
- [NumPy](https://numpy.org/) – numerical operations  
- [Matplotlib](https://matplotlib.org/) – plotting  
- [Seaborn](https://seaborn.pydata.org/) – visualizations  
- [Scikit-learn](https://scikit-learn.org/stable/) – K-Means clustering, preprocessing  


# Steps in the Project

Load Dataset
 - Import customer data (Mall_Customers.csv or other).

Preprocessing
 - Select relevant features (Annual Income, Spending Score).
 - Normalize features for fair clustering using StandardScaler.

 Elbow Method
 - Compute Within-Cluster-Sum-of-Squares (WCSS) for different K values.
 - Plot Elbow Curve to determine optimal number of clusters.

K-Means Clustering
 - Apply K-Means with chosen K.
 - Assign cluster labels to customers.

Visualization
 - Scatter plots of clusters with centroids.
 - Different colors for each cluster.

# Results

- Optimal clusters: usually K = 5 for Mall Customers dataset.

- Example segments found:

    - Cluster 1: High Income, High Spending (Premium Customers)

    - Cluster 2: Low Income, Low Spending (Budget-Conscious)

    - Cluster 3: High Income, Low Spending (Careful Spenders)

    - Cluster 4: Low Income, High Spending (Impulsive Buyers)

    - Cluster 5: Medium Income, Medium Spending

# Visualizations

- Elbow Plot – helps decide K.

- Scatter Plots – shows clusters and centroids.