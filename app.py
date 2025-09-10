import streamlit as st
import pandas as pd
import pickle
import os
import lightgbm
import numpy as np
import sys
import warnings

# Constants for file paths
CHECKPOINT_MODEL_FILE = 'checkpoints/model_trained.pkl'

# Add project root to path to allow importing from src
sys.path.append('.')
from src.models.model_trainer import ModelTrainer

st.set_page_config(page_title="Credit Default Risk Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    """
    Loads necessary artifacts for the app: model.
    Uses Streamlit's caching to load only once.
    """
    # Check if required files exist
    required_files = [CHECKPOINT_MODEL_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        st.error(
            f"Missing required files: {', '.join(missing_files)}. "
            "Please ensure these files are present."
        )
        st.stop()

    # Load the trained model
    with open(CHECKPOINT_MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['lgb_result']['models'][0] # Load model from lgb_result

    return model

def display_prediction_gauge(prediction_prob, title="Prediction Gauge"):
    """Displays a gauge-like meter for the prediction probability."""
    # Determine color based on risk
    if prediction_prob < 0.3:
        color = "green"
        risk_level = "Low Risk"
    elif prediction_prob < 0.6:
        color = "orange"
        risk_level = "Medium Risk"
    else:
        color = "red"
        risk_level = "High Risk"

    st.markdown(f"### {title}")
    st.markdown(
        f"""
        <div style="
            border: 2px solid {color}; 
            border-radius: 10px; 
            padding: 20px; 
            text-align: center; 
            background-color: #f0f2f6;
        ">
            <h2 style="color:{color}; margin:0;">{prediction_prob:.2%}</h2>
            <p style="margin:0;">Probability of Default</p>
            <h3 style="color:{color}; margin-top:10px;">{risk_level}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )



def new_applicant_entry(model):
    """UI for entering data for a completely new applicant."""
    st.header("Predict Risk for a New Applicant")

    with st.expander("⚠️ Important: Please Read About Prediction Accuracy"):
        st.warning(
            """
            **Note on Prediction Accuracy:**

            This prediction model relies heavily on an applicant's past credit history (e.g., from other banks or credit bureaus). 
            
            For a completely new applicant, this historical data is unavailable. This tool uses average, default values for these missing historical features.

            Therefore, the prediction generated here is an **estimate** based only on the information you provide in the sidebar. It may be less accurate than a prediction for a customer whose full financial history is available to the model.
            """
        )

    st.info("Please enter the applicant's details below. Other fields will be filled with sensible defaults.")

    st.sidebar.header("New Applicant Data")

    # Use more of the top important features that a user can provide
    editable_features = {
        'AMT_INCOME_TOTAL': 'Total Income',
        'AMT_CREDIT': 'Credit Amount of the loan',
        'AMT_ANNUITY': 'Loan Annuity',
        'DAYS_BIRTH': 'Age',
        'EXT_SOURCE_1': 'External Source 1 Score (0-1)',
        'EXT_SOURCE_2': 'External Source 2 Score (0-1)',
        'EXT_SOURCE_3': 'External Source 3 Score (0-1)'
    }

    # Get the full list of feature columns the model expects
    feature_cols = list(model.feature_name_) # Assuming LightGBM model has feature_name_ attribute

    # --- Create Default Values for ALL features ---
    # This is a simplification without test_features.csv
    # Identify numeric and categorical features based on common patterns or hardcoding
    known_categorical_features = ['CODE_GENDER', 'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'] # Example, adjust as needed

    numeric_cols = [f for f in feature_cols if f not in known_categorical_features]
    categorical_cols = [f for f in feature_cols if f in known_categorical_features]

    numeric_defaults = pd.Series({col: 0.0 for col in numeric_cols})
    categorical_defaults = pd.Series({col: "" for col in categorical_cols}) # Use empty string for categorical defaults
    
    default_values = pd.concat([numeric_defaults, categorical_defaults])

    # Create a dictionary to hold the new applicant's data from user input
    new_applicant_data = {}

    # Collect user input for the key features
    for feature, label in editable_features.items():
        # Special handling for age
        if feature == 'DAYS_BIRTH':
            age = st.sidebar.number_input(label=label, min_value=18, max_value=100, value=35, step=1, key='new_age')
            new_applicant_data[feature] = -age * 365 # Convert age in years to days and make it negative
            continue

        # Set appropriate defaults and ranges for other inputs
        if 'EXT_SOURCE' in feature:
            min_val, max_val, default_val, step = 0.0, 1.0, float(default_values.get(feature, 0.5)), 0.01
            format_str = "%.2f"
        else:
            min_val, max_val, default_val, step = 0.0, None, float(default_values.get(feature, 0)), 1000.0
            format_str = "%.2f"

        user_input = st.sidebar.number_input(
            label=label, min_value=min_val, max_value=max_val, value=default_val,
            step=step, format=format_str, key=f"new_{feature}"
        )
        new_applicant_data[feature] = user_input

    if st.button("Predict Default Risk"):
        # Create the full feature vector
        new_vector_series = default_values.copy()
        # Overwrite with user input
        for feature, value in new_applicant_data.items():
            new_vector_series[feature] = value
        
        # --- Add Simple Feature Engineering based on user input ---
        # This replicates some of the logic from the feature engineering script
        # for the most important features.
        
        # Calculate CREDIT_TERM
        # Adding a small epsilon to avoid division by zero
        annuity = new_vector_series.get('AMT_ANNUITY', 1)
        credit = new_vector_series.get('AMT_CREDIT', 0)
        if 'CREDIT_TERM' in new_vector_series:
            new_vector_series['CREDIT_TERM'] = credit / (annuity + 1e-6)

        # Calculate EXT_SOURCES derived features
        ext_sources = [
            new_vector_series.get('EXT_SOURCE_1', np.nan),
            new_vector_series.get('EXT_SOURCE_2', np.nan),
            new_vector_series.get('EXT_SOURCE_3', np.nan)
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if 'EXT_SOURCES_MEAN' in new_vector_series:
                new_vector_series['EXT_SOURCES_MEAN'] = np.nanmean(ext_sources)
            if 'EXT_SOURCES_MIN' in new_vector_series:
                new_vector_series['EXT_SOURCES_MIN'] = np.nanmin(ext_sources)
            if 'EXT_SOURCES_MAX' in new_vector_series:
                new_vector_series['EXT_SOURCES_MAX'] = np.nanmax(ext_sources)

        # Convert to DataFrame
        new_applicant_df = pd.DataFrame(new_vector_series).T
        
        # --- PREPROCESSING FOR PREDICTION ---
        # Ensure all feature columns are present and in the correct order
        # This line is removed: for col in new_applicant_df.columns: if col in test_features.columns: new_applicant_df[col] = new_applicant_df[col].astype(test_features[col].dtype)

        new_applicant_df = new_applicant_df.replace([np.inf, -np.inf], np.nan)
        # Fill any remaining NaNs with the default values again to be safe
        new_applicant_df = new_applicant_df.fillna(default_values)

        # Prepare data for prediction
        modified_vector = new_applicant_df[feature_cols].values.reshape(1, -1)

        # Predict with the modified data
        new_prediction = model.predict(modified_vector)[0]

        # Display results
        display_prediction_gauge(new_prediction, "New Applicant's Predicted Risk")

def main():
    """Main function to run the Streamlit app."""
    st.title("🏠 Home Credit Default Risk Predictor")

    # Load all necessary artifacts
    model = load_artifacts()

    st.sidebar.info("Prediction Model: **LightGBM**")

    new_applicant_entry(model)


if __name__ == "__main__":
    main()
