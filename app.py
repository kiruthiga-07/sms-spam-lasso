import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Title
# -----------------------------
st.title("📩 SMS Spam Classification using Lasso + Logistic Regression")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

df = load_data()

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

st.subheader("📊 Dataset Preview")
st.write(df.head())

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

total_features = X.shape[1]
st.write(f"🔹 Total TF-IDF Features: {total_features}")

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Lasso Feature Selection Function
# -----------------------------
def lasso_feature_selection(alpha):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train.toarray(), y_train)

    coef = lasso.coef_
    selected = coef != 0

    non_zero = np.sum(selected)
    zero = np.sum(coef == 0)

    return selected, non_zero, zero

# -----------------------------
# Alpha = 0.1 (Main Model)
# -----------------------------
selected_01, non_zero_01, zero_01 = lasso_feature_selection(0.1)

st.subheader("📉 Lasso Feature Selection (alpha = 0.1)")
st.write(f"✅ Selected features: {non_zero_01}")
st.write(f"❌ Eliminated features: {zero_01}")

# -----------------------------
# Alpha Comparison
# -----------------------------
_, non_zero_001, _ = lasso_feature_selection(0.01)
_, non_zero_1, _ = lasso_feature_selection(1)

st.subheader("📊 Alpha Comparison")
st.write(f"Alpha 0.01 → {non_zero_001} features selected")
st.write(f"Alpha 0.1  → {non_zero_01} features selected")
st.write(f"Alpha 1    → {non_zero_1} features selected")

# -----------------------------
# Feature Reduction %
# -----------------------------
reduction = ((total_features - non_zero_01) / total_features) * 100
st.write(f"📉 Feature Reduction: {reduction:.2f}%")

# -----------------------------
# Reduce Features
# -----------------------------
X_train_sel = X_train[:, selected_01]
X_test_sel = X_test[:, selected_01]

# -----------------------------
# Train Logistic Regression
# -----------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_sel, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = clf.predict(X_test_sel)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# -----------------------------
# Show Sample Predictions
# -----------------------------
st.subheader("📋 Sample Predictions")

sample_df = df.iloc[:10].copy()
sample_vec = vectorizer.transform(sample_df['message'])
sample_vec_sel = sample_vec[:, selected_01]
sample_pred = clf.predict(sample_vec_sel)

sample_df['Predicted'] = sample_pred
sample_df['Predicted'] = sample_df['Predicted'].map({0: 'ham', 1: 'spam'})

st.write(sample_df[['message', 'Predicted']])

# -----------------------------
# User Input Prediction
# -----------------------------
st.subheader("✉️ Predict Your Own SMS")

user_input = st.text_area("Enter SMS message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        input_vec = vectorizer.transform([user_input])
        input_vec_sel = input_vec[:, selected_01]

        pred = clf.predict(input_vec_sel)[0]

        if pred == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Ham (Not Spam)")
