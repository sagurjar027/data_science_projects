import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Churn Prediction Pro", layout="centered")
st.title("📞 Customer Churn Prediction System")
# -------------------- LOAD ASSETS --------------------
@st.cache_resource
def load_assets():
    try:
        rf_model = joblib.load('rf_model.joblib')
        xgb_model = joblib.load('xgb_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return rf_model, xgb_model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

rf_model, xgb_model, scaler = load_assets()

# -------------------- MAIN APP --------------------
def main():
    st.markdown("---")

    # -------------------- SIDEBAR --------------------
    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.selectbox(
            "Choose Model",
            ["Random Forest (High Recall)", "XGBoost (Balanced)"]
        )

        if model_choice == "Random Forest (High Recall)":
            st.info("🔍 Focus: Maximum churn detection (High Recall)")
        else:
            st.info("⚖️ Focus: Balanced precision & recall")

        st.markdown("---")
        st.markdown("## 🤖 About Models")
        st.info("""
        🔹 Random Forest → Higher Recall (~79%)  
        🔹 XGBoost → Balanced performance  

        ⚙️ Threshold tuning:
        - RF → 0.4  
        - XGB → 0.6  
        """)

    # -------------------- INPUT UI --------------------
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Gender', ['female', 'male'])
        SeniorCitizen = st.selectbox('Senior Citizen', [0, 1])
        Partner = st.selectbox('Partner', ['Yes', 'No'])
        Dependents = st.selectbox('Dependents', ['Yes', 'No'])
        tenure = st.slider('Tenure (Months)', 0, 72, 1)
        PhoneService = st.selectbox('Phone Service', ['Yes', 'No'])
        MultipleLines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])

    with col2:
        InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        OnlineBackup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        DeviceProtection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        TechSupport = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        StreamingMovies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

    st.markdown("### Billing & Contract")
    c3, c4, c5 = st.columns(3)

    with c3:
        Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    with c4:
        PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
    with c5:
        PaymentMethod = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])

    MonthlyCharges = st.number_input('Monthly Charges', 0.0, 150.0, 29.85)
    TotalCharges = st.number_input('Total Charges', 0.0, 9000.0, 29.85)

    # -------------------- MAPPING & PREPROCESSING --------------------
    
    # 1. Mappings
    mapping = {
        "gender": {"male": 1, "female": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "MultipleLines": {"Yes": 2, "No": 0, "No phone service": 1},
        "InternetService": {"Fiber optic": 2, "DSL": 1, "No": 0},
        "OnlineSecurity": {"Yes": 2, "No": 0, "No internet service": 1},
        "OnlineBackup": {"Yes": 2, "No": 0, "No internet service": 1},
        "DeviceProtection": {"Yes": 2, "No": 0, "No internet service": 1},
        "TechSupport": {"Yes": 2, "No": 0, "No internet service": 1},
        "StreamingTV": {"Yes": 2, "No": 0, "No internet service": 1},
        "StreamingMovies": {"Yes": 2, "No": 0, "No internet service": 1},
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
        "PaperlessBilling": {"Yes": 1, "No": 0},
        "PaymentMethod": {
            "Electronic check": 0, "Mailed check": 1,
            "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
        }
    }

    # 2. Build DataFrame
    input_dict = {
        'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner, 
        'Dependents': Dependents, 'tenure': tenure, 'PhoneService': PhoneService, 
        'MultipleLines': MultipleLines, 'InternetService': InternetService, 
        'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup, 
        'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport, 
        'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies, 
        'Contract': Contract, 'PaperlessBilling': PaperlessBilling, 
        'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges, 
        'TotalCharges': TotalCharges
    }
    
    final_df = pd.DataFrame([input_dict]).replace(mapping)

    # 3. Apply Scaler to correct features (as per your ValueError)
    features_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    final_df[features_to_scale] = scaler.transform(final_df[features_to_scale])

    # 4. Final Column Order Check (Crucial for ML models)
    column_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
        'MonthlyCharges', 'TotalCharges'
    ]
    final_input = final_df[column_order]

    st.markdown("---")

    # -------------------- PREDICTION --------------------
    if st.button("🔍 Analyze Customer Churn Risk", use_container_width=True):
        if rf_model is None or xgb_model is None:
            st.error("Models not loaded. Check your .pkl files.")
            return

        # Select Model and Threshold
        if model_choice == "Random Forest (High Recall)":
            model = rf_model
            custom_threshold = 0.4
        else:
            model = xgb_model
            custom_threshold = 0.6

        # Get Probability
        prob = model.predict_proba(final_input)[0][1]

        st.markdown("## 📊 Prediction Result")
        st.write(f"Churn Probability: **{prob:.2%}**")

        # Visual Result based on Probability
        if prob >= custom_threshold:
            st.error(f"🚨 High Risk (Probability > {custom_threshold})")
        else:
            st.success(f"✅ Low Risk (Probability < {custom_threshold})")

        # Business Recommendation Logic
        st.markdown("### 💡 Business Recommendation")
        if prob > 0.7:
            st.write("- **Immediate Action:** Offer high-value retention discount.")
            st.write("- **Personal Touch:** Assign a success manager to call.")
        elif prob > 0.4:
            st.write("- **Engagement:** Send educational content about unused features.")
            st.write("- **Incentive:** Offer a small credit on the next bill.")
        else:
            st.write("- **Stable:** Maintain standard high-quality service.")
            st.write("- **Upsell:** Good candidate for premium feature trials.")

    st.markdown("---")
    st.markdown("<div style='text-align:center;'>Developed by <b>Sahil Kasana</b> 🚀</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()