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
TEST_FEATURES_FILE = 'data/processed/test_features.csv'
SUBMISSION_FILE = 'checkpoints/submission_created.pkl'
FEATURE_IMPORTANCE_FILE = 'data/processed/feature_importance.csv'

# Add project root to path to allow importing from src
sys.path.append('.')
from src.models.model_trainer import ModelTrainer

st.set_page_config(page_title="Credit Default Risk Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    """
    Loads all necessary artifacts for the app: model, test data, predictions, and feature importances.
    Uses Streamlit's caching to load only once.
    """
    # Check if all required files exist
    required_files = [CHECKPOINT_MODEL_FILE, TEST_FEATURES_FILE, SUBMISSION_FILE, FEATURE_IMPORTANCE_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        st.error(
            f"Missing required files: {', '.join(missing_files)}. "
            "Please run the `quick_start.py` script to generate these files before starting the app."
        )
        st.stop()

    # Load the trained model and trainer object
    with open(CHECKPOINT_MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)
    trainer = model_data['trainer']
    model = model_data['lgb_result']['models'][0] # Load model from lgb_result

    # Load test features
    test_features = pd.read_csv(TEST_FEATURES_FILE)

    # Load submission data to get pre-computed predictions
    with open(SUBMISSION_FILE, 'rb') as f:
        submission_data = pickle.load(f)
    
    predictions_df = pd.DataFrame({
        'SK_ID_CURR': submission_data['test_ids'],
        'PREDICTION': submission_data['predictions']
    })

    # Load feature importances
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)

    return trainer, model, test_features, predictions_df, feature_importance

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

def analyze_existing_applicant(trainer, model, test_features, predictions_df, feature_importance):
    """UI for analyzing an existing applicant."""
    st.header("Analyze an Existing Applicant's Default Risk")
    
    # Select an applicant
    applicant_id = st.selectbox(
        "Select Applicant ID (from test set)",
        options=test_features['SK_ID_CURR'].unique()
    )

    if applicant_id:
        # Get applicant data and prediction
        applicant_data = test_features[test_features['SK_ID_CURR'] == applicant_id]
        prediction_prob = predictions_df[predictions_df['SK_ID_CURR'] == applicant_id]['PREDICTION'].iloc[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            display_prediction_gauge(prediction_prob, "Credit Default Risk")

        with col2:
            st.subheader("Key Applicant Information")
            # Display a subset of important and readable features
            display_features = [
                'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CODE_GENDER'
            ]
            # Filter for columns that actually exist in the dataframe
            display_features = [f for f in display_features if f in applicant_data.columns]
            st.dataframe(applicant_data[display_features].T.rename(columns={applicant_data.index[0]: 'Value'}))

        # Feature Importance/Contribution
        st.subheader("Prediction Explanation (Top 10 Features)")
        st.write("These are the global feature importances from the model, indicating which factors are generally most influential.")
        st.dataframe(feature_importance.head(10))

def new_applicant_simulation(trainer, model, test_features, predictions_df):
    """UI for simulating a new applicant based on an existing one."""
    st.header("New Applicant Simulation (What-If Analysis)")
    st.info(
        "Select a baseline applicant from the test set, then modify their key financial details below. "
        "The model will predict the new risk based on your changes."
    )

    baseline_id = st.selectbox(
        "Select a Baseline Applicant ID to modify",
        options=test_features['SK_ID_CURR'].unique(),
        key='baseline_selector'
    )

    if not baseline_id:
        st.stop()

    # Get original data and prediction
    original_data = test_features[test_features['SK_ID_CURR'] == baseline_id].iloc[0]
    original_prediction = predictions_df[predictions_df['SK_ID_CURR'] == baseline_id]['PREDICTION'].iloc[0]

    st.sidebar.header("Modify Applicant Data")
    
    # Features the user can edit
    editable_features = {
        'AMT_INCOME_TOTAL': 'Total Income',
        'AMT_CREDIT': 'Credit Amount of the loan',
        'AMT_ANNUITY': 'Loan Annuity',
        'EXT_SOURCE_2': 'External Source 2 Score',
        'EXT_SOURCE_3': 'External Source 3 Score'
    }

    modified_data = original_data.copy()
    
    for feature, label in editable_features.items():
        if feature in modified_data:
            min_val = 0.0
            # Use numpy types for compatibility with Streamlit widgets
            default_val = float(original_data[feature]) if not pd.isna(original_data[feature]) else 0.0
            
            modified_data[feature] = st.sidebar.number_input(
                label=label,
                value=default_val,
                min_value=min_val,
                step=1000.0,
                format="%.2f"
            )

    # --- PREPROCESSING FOR PREDICTION ---
    # Convert to DataFrame for easier processing and to align with training preprocessing
    modified_df = pd.DataFrame(modified_data).T
    modified_df = modified_df.replace([np.inf, -np.inf], np.nan)

    # Get column types from the loaded test_features dataframe
    numeric_cols = test_features.select_dtypes(include=np.number).columns
    
    # Use medians from the loaded test_features as a stand-in for training data medians
    numeric_medians = test_features[numeric_cols].median()

    # Fill NaN values in the modified data
    modified_df[numeric_cols] = modified_df[numeric_cols].fillna(numeric_medians)

    # It's assumed that categorical features do not have NaNs in the processed test data,
    # as they are less likely to be introduced by the simulation controls.
    # If they could, a similar mode-based filling would be needed.

    # Prepare data for prediction
    feature_cols = [col for col in test_features.columns if col not in ['SK_ID_CURR', 'TARGET']]
    
    # Ensure all feature columns are present and in the correct order
    modified_vector = modified_df[feature_cols].values.reshape(1, -1)

    # Predict with the modified data
    new_prediction = model.predict(modified_vector)[0]

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        display_prediction_gauge(original_prediction, "Baseline Applicant Risk")
        st.caption(f"Based on original data for applicant {baseline_id}")

    with col2:
        display_prediction_gauge(new_prediction, "Simulated Applicant Risk")
        st.caption("Based on your modifications in the sidebar")

    # Show changes
    st.subheader("Summary of Changes")
    changes = []
    for feature in editable_features:
        if feature in modified_data:
            original_val = original_data[feature]
            new_val = modified_data[feature]
            if original_val != new_val:
                changes.append({
                    'Feature': editable_features[feature],
                    'Original Value': original_val,
                    'New Value': new_val
                })
    if changes:
        st.table(pd.DataFrame(changes))
    else:
        st.write("No changes made from the baseline applicant.")

def new_applicant_entry(model, test_features):
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
    feature_cols = [col for col in test_features.columns if col not in ['SK_ID_CURR', 'TARGET']]
    
    # --- Create Default Values for ALL features ---
    numeric_cols = test_features[feature_cols].select_dtypes(include=np.number).columns
    categorical_cols = test_features[feature_cols].select_dtypes(exclude=np.number).columns
    numeric_defaults = test_features[numeric_cols].median()
    categorical_defaults = pd.Series({col: test_features[col].mode(dropna=True)[0] if not test_features[col].mode(dropna=True).empty else "" for col in categorical_cols})
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
        # Ensure dtypes match the original dataframe
        for col in new_applicant_df.columns:
            if col in test_features.columns:
                new_applicant_df[col] = new_applicant_df[col].astype(test_features[col].dtype)

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
    trainer, model, test_features, predictions_df, feature_importance = load_artifacts()

    # Sidebar for mode selection
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Analyze Existing Applicant", "New Applicant Simulation", "Predict for a New Applicant"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Prediction Model: **LightGBM**")

    if app_mode == "Analyze Existing Applicant":
        analyze_existing_applicant(trainer, model, test_features, predictions_df, feature_importance)
    elif app_mode == "New Applicant Simulation":
        new_applicant_simulation(trainer, model, test_features, predictions_df)
    elif app_mode == "Predict for a New Applicant":
        new_applicant_entry(model, test_features)


if __name__ == "__main__":
    main()
