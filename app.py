import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.title("App Users Segmentation")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    features = ["Average Screen Time", "Average Spent on App (INR)", "Ratings", "Last Visited Minutes"]
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = pickle.load(open("app_users_kmeans.pkl", "rb"))
    df["cluster"] = model.predict(X_scaled)
    
    st.write(df.head())
    st.write("Clusters Distribution")
    st.bar_chart(df["cluster"].value_counts())
