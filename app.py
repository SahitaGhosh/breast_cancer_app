import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Title
st.title("🔬 Breast Cancer Classifier App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Preprocessing
    if 'Unnamed: 32' in df.columns:
        df.drop(['Unnamed: 32'], axis=1, inplace=True)
    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)
    
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    st.subheader("🧾 Dataset Preview")
    st.dataframe(df.head())

    # Feature selection
    prediction_vars = [
        "radius_mean", "perimeter_mean", "area_mean",
        "compactness_mean", "concavity_mean", "concave points_mean",
        "radius_worst", "perimeter_worst", "compactness_worst"
    ]

    X = df[prediction_vars]
    y = df['diagnosis']

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state=1)

    # Model selection
    model_name = st.selectbox("Choose Model", ("Random Forest", "KNN", "SVM"))

    if model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    else:
        model = SVC()

    if st.button("Train and Predict"):
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)

        acc = accuracy_score(test_y, predictions)
        prec = precision_score(test_y, predictions)
        rec = recall_score(test_y, predictions)

        st.success(f"✅ Accuracy: {acc:.2f}")
        st.info(f"📊 Precision: {prec:.2f}")
        st.warning(f"📈 Recall: {rec:.2f}")

        # Confusion matrix
        cm = confusion_matrix(test_y, predictions)
        st.subheader("🧩 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'], ax=ax)
        st.pyplot(fig)

        # PCA Visualization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_pca['Target'] = y.values

        st.subheader("🔎 PCA (2D) Visualization")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Target", palette='Set2', alpha=0.7, ax=ax2)
        st.pyplot(fig2)
