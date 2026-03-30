import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title
st.title("📩 SMS Spam Classification using Lasso (Feature Selection)")

# Load dataset
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

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

total_features = X.shape[1]

st.write(f"🔹 Total TF-IDF Features: {total_features}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train Lasso
def train_lasso(alpha):
    model = Lasso(alpha=alpha)
    model.fit(X_train.toarray(), y_train)

    coef = model.coef_
    non_zero = np.sum(coef != 0)
    zero = np.sum(coef == 0)

    return model, non_zero, zero

# Alpha = 0.1
model_01, non_zero_01, zero_01 = train_lasso(0.1)

st.subheader("📉 Lasso (alpha = 0.1)")
st.write(f"Non-zero features: {non_zero_01}")
st.write(f"Eliminated features: {zero_01}")

# Alpha comparison
_, non_zero_001, _ = train_lasso(0.01)
_, non_zero_1, _ = train_lasso(1)

st.subheader("📊 Feature Selection Comparison")
st.write(f"Alpha 0.01 → Selected features: {non_zero_001}")
st.write(f"Alpha 0.1 → Selected features: {non_zero_01}")
st.write(f"Alpha 1 → Selected features: {non_zero_1}")

# Reduction %
reduction = ((total_features - non_zero_01) / total_features) * 100
st.write(f"📉 Feature Reduction: {reduction:.2f}%")

# Predictions on test data
y_pred = model_01.predict(X_test.toarray())
y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]

accuracy = accuracy_score(y_test, y_pred_class)

st.subheader("📈 Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# User input prediction
st.subheader("✉️ Predict Custom SMS")

user_input = st.text_area("Enter SMS message:")

if st.button("Predict"):
    input_vec = vectorizer.transform([user_input])
    pred = model_01.predict(input_vec.toarray())[0]

    if pred > 0.5:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Ham (Not Spam)")
