import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ Customer Segmentation Using K-Means")
st.markdown("### Retail Store Customer Analysis")

# Load data
df = pd.read_csv("Mall_Customers.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Feature selection
st.subheader("🔧 Select Features for Clustering")
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Choose number of clusters
k = st.slider("Select Number of Clusters (K)", 2, 10, 5)

# Train model
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# Visualization
st.subheader("📈 Customer Clusters Visualization")

fig, ax = plt.subplots()
scatter = ax.scatter(
    features.iloc[:, 0],
    features.iloc[:, 1],
    c=df['Cluster'],
)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segments")

st.pyplot(fig)

cluster_summary = df.groupby('Cluster')[[
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]].mean().round(2)
cluster_names = {
    0: "Average Income – Average Spending",
    1: "High Income – High Spending",
    2: "Low Income – High Spending",
    3: "High Income – Low Spending",
    4: "Low Income – Low Spending"
}

cluster_summary['Customer Segment'] = cluster_summary.index.map(cluster_names)

st.subheader("🧠 Cluster Insights")
st.dataframe(cluster_summary)


