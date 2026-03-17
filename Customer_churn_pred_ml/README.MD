# 📞 Customer Churn Prediction System

An end-to-end Machine Learning project to predict customer churn in the telecom industry using multiple models and business-driven optimization.

---

## 🚀 Project Overview

Customer churn directly impacts revenue. This project helps identify customers likely to leave, enabling proactive retention strategies.

---

## 📊 Exploratory Data Analysis (EDA) please visit
https://github.com/sagurjar027/data_science_projects/blob/sagurjar027-patch-1/Customer_churn_pred_ml/Classification-%20Telecom%20Churn.ipynb
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


## 📁 Project Structure

```text
Customer_churn_pred_ml/
├── Classification- Telecom Churn.ipynb   # Complete EDA, preprocessing, and model building
├── app.py                                # Streamlit web application script
├── Telecom_Churn.csv                     # Raw dataset
├── requirements.txt                      # Project dependencies
├── rf_model.joblib                       # Serialized Random Forest model
├── xgb_model.joblib                      # Serialized XGBoost model
├── scaler.joblib                         # Saved data scaler for preprocessing
├── data                                  # Additional data artifacts
└── README.MD                             # Project documentation


1. Clone the repository and navigate to the project directory:

Bash
git clone [https://github.com/sagurjar027/data_science_projects.git](https://github.com/sagurjar027/data_science_projects.git)
cd data_science_projects/Customer_churn_pred_ml

2. Install the required dependencies:

Bash
pip install -r requirements.txt
3. Run the Streamlit app:

Bash
streamlit run app.py


👨‍💻 Author
Sahil Kasana
Second-Year Computer Science Student & Data Science Enthusiast
Passionate about building end-to-end Machine Learning pipelines, Generative AI, and solving real-world business problems through data.

GitHub: @sagurjar027

