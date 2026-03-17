# 📞 Customer Churn Prediction System

An end-to-end Machine Learning project to predict customer churn in the telecom industry using multiple models and business-driven optimization.

---

## 🚀 Project Overview

Customer churn directly impacts revenue. This project helps identify customers likely to leave, enabling proactive retention strategies.

---

## 📊 Exploratory Data Analysis (EDA)

### 🔹 Churn Distribution
![Churn Distribution](images/churn_distribution.png)

### 🔹 Gender vs Churn
![Gender vs Churn](images/gender_churn.png)

### 🔹 Contract Type vs Churn
![Contract vs Churn](images/contract_churn.png)

### 🔹 Monthly Charges vs Churn
![Monthly Charges](images/monthly_charges.png)

---

## 🧠 Key Insights

- 📉 Customers with **month-to-month contracts** churn the most  
- 💰 Higher **monthly charges → higher churn probability**  
- ⏳ Customers with **longer tenure are more loyal**  
- 📡 Fiber optic users show higher churn  

---

## ⚙️ Models Used

### 🔹 Random Forest (High Recall Model)
- Focus: Detect maximum churn customers  
- Recall: ~79%  
- Threshold: 0.4  

### 🔹 XGBoost (Balanced Model)
- Focus: Precision + Recall balance  
- Recall: ~66%  
- Threshold: 0.6  

---

## 🎯 Model Optimization

Instead of default threshold (0.5), **threshold tuning** was applied:

| Model | Threshold | Recall | Precision |
|------|----------|--------|----------|
| Random Forest | 0.4 | ~0.79 | ~0.52 |
| XGBoost | 0.6 | ~0.66 | ~0.60 |

👉 Business goal: **maximize recall to reduce churn loss**

---

## ⚙️ ML Pipeline

1. Data Cleaning & Preprocessing  
2. Feature Encoding (Manual Mapping)  
3. Feature Scaling (Numerical Only)  
4. Model Training  
5. Threshold Tuning  
6. Model Comparison  
7. Deployment using Streamlit  

---

## 💻 Streamlit App Features

- 🔍 Real-time churn prediction  
- ⚙️ Model selection (RF / XGB)  
- 📊 Probability-based risk classification  
- 💡 Business recommendations  

---

## 📂 Project Structure
