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
# Lasso Feature Selection
# -----------------------------
def lasso_feature_selection(alpha):
    lasso = Lasso(alpha=alpha, max_iter=5000)
    lasso.fit(X_train.toarray(), y_train)

    coef = lasso.coef_
    selected = coef != 0

    non_zero = np.sum(selected)
    zero = np.sum(coef == 0)

    return selected, non_zero, zero

# -----------------------------
# MAIN MODEL (alpha = 0.001 FIXED)
# -----------------------------
selected_main, non_zero_main, zero_main = lasso_feature_selection(0.001)

st.subheader("📉 Lasso Feature Selection (alpha = 0.001)")
st.write(f"✅ Selected features: {non_zero_main}")
st.write(f"❌ Eliminated features: {zero_main}")

# -----------------------------
# Safety Check (VERY IMPORTANT)
# -----------------------------
if non_zero_main == 0:
    st.error("⚠️ All features eliminated. Reduce alpha further (try 0.0001).")
    st.stop()

# -----------------------------
# Alpha Comparison
# -----------------------------
_, nz_0001, _ = lasso_feature_selection(0.0001)
_, nz_001, _ = lasso_feature_selection(0.001)
_, nz_01, _ = lasso_feature_selection(0.01)

st.subheader("📊 Alpha Comparison")
st.write(f"Alpha 0.0001 → {nz_0001} features")
st.write(f"Alpha 0.001  → {nz_001} features")
st.write(f"Alpha 0.01   → {nz_01} features")

# -----------------------------
# Feature Reduction %
# -----------------------------
reduction = ((total_features - non_zero_main) / total_features) * 100
st.write(f"📉 Feature Reduction: {reduction:.2f}%")

# -----------------------------
# Reduce Features
# -----------------------------
X_train_sel = X_train[:, selected_main]
X_test_sel = X_test[:, selected_main]

# -----------------------------
# Train Logistic Regression
# -----------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_sel, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = clf.predict(X_test_sel)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# -----------------------------
# Sample Predictions
# -----------------------------
st.subheader("📋 Sample Predictions")

sample_df = df.iloc[:10].copy()
sample_vec = vectorizer.transform(sample_df['message'])
sample_vec_sel = sample_vec[:, selected_main]
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
        input_vec_sel = input_vec[:, selected_main]

        pred = clf.predict(input_vec_sel)[0]

        if pred == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Ham (Not Spam)")
