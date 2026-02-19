🧠 ASD Screening using Deep Neural Networks

Deep Learning–based medical screening system to predict the likelihood of Autism Spectrum Disorder (ASD) using behavioral and demographic data.

This project demonstrates an end-to-end deep learning pipeline — from data cleaning and leakage prevention to model deployment.

📌 Table of Contents

Project Overview

Key Results

Dataset Description

Technical Implementation

Model Architecture

Training & Evaluation

Project Structure

How to Run

Future Improvements

About Me

🚀 Project Overview

Early detection of Autism Spectrum Disorder (ASD) is crucial for timely intervention.

This project builds a binary classification Deep Neural Network (YES/NO) to assist in ASD screening using structured behavioral questionnaire data.

Goals:

Build a reliable Deep Learning classifier

Prevent data leakage

Optimize recall (important for medical screening)

Ensure strong generalization without overfitting

📊 Key Results

Accuracy: ~98%

Recall (Sensitivity): 97.2%

False Negatives: 1 case

False Positives: 2 cases

Overfitting: None observed (smooth validation convergence)

Confusion Matrix
Predicted \ Actual	NO	YES
NO	103 (TN)	1 (FN)
YES	2 (FP)	35 (TP)

The model minimizes false negatives — critical for medical screening systems.

📂 Dataset Description

The dataset includes:

Behavioral screening scores

Demographic information (gender, ethnicity, etc.)

Binary ASD diagnosis label

Data Cleaning Performed

Removed data leakage column (result) that directly computed final label

Corrected extreme outlier (age = 383)

Handled missing values

One-Hot Encoded categorical variables

Applied feature scaling using StandardScaler

🛠 Technical Implementation
1️⃣ Data Preprocessing

Outlier correction

One-Hot Encoding

Feature Scaling

Train-Test Split

2️⃣ NaN Loss Debugging

During early training, model loss became NaN.

Resolved by:

Scaling input features

Cleaning inconsistent values

Reducing learning rate

🏗 Model Architecture

Framework: TensorFlow / Keras
Model Type: Feedforward Deep Neural Network

Architecture:

Input Layer

Dense Layer (ReLU)

Dense Layer (ReLU)

Output Layer (Sigmoid)

Loss Function: Binary Crossentropy
Optimizer: Adam
Evaluation Metric: Accuracy, Recall

The model captures nonlinear behavioral interactions for improved predictive performance.

📈 Training & Evaluation

Binary Classification Task

Stratified train-test split

Performance evaluated using:

Accuracy

Recall

Confusion Matrix

Checked for overfitting using validation loss curves


💻 How to Run
Clone Repository
git clone https://github.com/sagurjar027/data_science_projects/edit/main/Deep%20Learning/
cd ASD-Screening-DL

Install Dependencies
pip install -r requirements.txt

Run Streamlit App
streamlit run app.py


🧪 Future Improvements
Hyperparameter tuning

Cross-validation

SHAP for model explainability

Cloud deployment (AWS / GCP)

Clinical validation on larger datasets



👤 About Me

Sahil Kasana
Second-year Computer Science student
Aspiring Data Scientist

Interested in building AI systems that solve real-world problems in healthcare, finance, and analytics

