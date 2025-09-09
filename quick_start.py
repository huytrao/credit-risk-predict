#!/usr/bin/env python3
"""
Home Credit Default Risk - 快速开始脚本 (优化版)

the code covers the following steps:
1. fist step is data loading
2. second step is feature engineering
3. third step is model training
4. fourth step is creating submission file

you can run the script as follows:
python quick_start.py [--force-restart] [--skip-to STEP]

arguments:
--force-restart: that's force restart all steps
--skip-to STEP: that's skip to the specified step (1-5)

"""

import os
import sys
import warnings
import argparse
import pandas as pd
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# the project path
sys.path.append('.')

from src.data.data_loader import HomeCreditDataLoader, reduce_memory_usage
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
import sys; print(sys.path)
from src.utils.visualization import DataVisualizer

# the checkpoint files
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_FILES = {
    'data_loaded': f'{CHECKPOINT_DIR}/data_loaded.pkl',
    'features_created': f'{CHECKPOINT_DIR}/features_created.pkl',
    'model_trained': f'{CHECKPOINT_DIR}/model_trained.pkl',
    'submission_created': f'{CHECKPOINT_DIR}/submission_created.pkl'
}

def create_checkpoint_dir():
    """the checkpoint directory"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(checkpoint_name, data):
    """the checkpoint"""
    create_checkpoint_dir()
    with open(CHECKPOINT_FILES[checkpoint_name], 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 the checkpoint '{checkpoint_name}' saved: {checkpoint_name}")

def load_checkpoint(checkpoint_name):
    """the checkpoint"""
    if os.path.exists(CHECKPOINT_FILES[checkpoint_name]):
        with open(CHECKPOINT_FILES[checkpoint_name], 'rb') as f:
            return pickle.load(f)
    return None

def checkpoint_exists(checkpoint_name):
    """ check the checkpoint"""
    return os.path.exists(CHECKPOINT_FILES[checkpoint_name])

def clear_checkpoints():
    """the checkpoint"""
    for checkpoint_file in CHECKPOINT_FILES.values():
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
    print("🗑️ All checkpoints cleared")

def check_data_files():
    """check the data files"""
    required_files = [
        'application_train.csv',
        'application_test.csv',
        'bureau.csv',
        'bureau_balance.csv',
        'previous_application.csv',
        'POS_CASH_balance.csv',
        'credit_card_balance.csv',
        'installments_payments.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(f'data/raw/{file}'):
            missing_files.append(file)
    
    return missing_files

def main():
    """the main function"""
    # the argument parser
    parser = argparse.ArgumentParser(description='Home Credit Default Risk - the quick start script (optimized)')
    parser.add_argument('--force-restart', action='store_true', help='Force restart all steps')
    parser.add_argument('--skip-to', type=int, choices=[1,2,3,4,5], help='Skip to the specified step')
    args = parser.parse_args()

        print("Home Credit Default Risk - the quick start (optimized)")
    print("=" * 50)
    
    # Handle force restart
    if args.force_restart:
        clear_checkpoints()
        print("🔄 Force restart mode, all checkpoints cleared")

    # Check data files
    missing_files = check_data_files()
    if missing_files:
        print(" Wrong data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📁 Please download the Kaggle dataset to the data/raw/ directory")
        return

    # Determine start step
    start_step = args.skip_to if args.skip_to else 1
    
    # Auto-detect completed steps
    if not args.skip_to and not args.force_restart:
        if checkpoint_exists('submission_created'):
            print("✅ All steps already completed.")
            print("💡 Use --force-restart to run again, or --skip-to N to skip to a specific step")
            return
        elif checkpoint_exists('model_trained'):
            start_step = 4
            print("🔍 Detected that model training is complete, starting from step 4...")
        elif checkpoint_exists('features_created'):
            start_step = 3
            print("🔍 Detected that feature engineering is complete, starting from step 3...")
        elif checkpoint_exists('data_loaded'):
            start_step = 2
            print("🔍 Detected that data loading is complete, starting from step 2...")

    print(f"🚀 Starting execution from step {start_step}...")
    print("=" * 50)

    # Initialize variables
    datasets = None
    train_features = None
    test_features = None
    lgb_result = None
    
    # 1. complete data loading
    if start_step <= 1:
        print("\n📊 Step 1: Data Loading")
        print("-" * 30)
        
        data_loader = HomeCreditDataLoader(data_path='data/raw')
        datasets = data_loader.load_all_data()
        
        if not datasets or 'application_train' not in datasets or datasets['application_train'] is None:
            print("❌ Data loading failed, please check the data files")
            return

        print("✅ Data loading completed!")
        data_loader.get_data_info()

        # Save checkpoint
        save_checkpoint('data_loaded', datasets)
    else:
        print("\n⏭️  Skip Step 1: Data Loading (Using Checkpoint)")
        datasets = load_checkpoint('data_loaded')
        if datasets is None:
            print("❌ Unable to load data checkpoint, please use --force-restart")
            return

    # 2. Feature Engineering
    if start_step <= 2:
        print("\n🔧 Step 2: Feature Engineering")
        print("-" * 30)
        
        feature_engineer = FeatureEngineer()
        
        try:
            train_features, test_features = feature_engineer.create_all_features(datasets)

            # Memory optimization
            print("\n🔧 Optimizing memory usage...")
            train_features = reduce_memory_usage(train_features)
            test_features = reduce_memory_usage(test_features)

            print("✅ Feature engineering completed!")
            print(f"   Training set shape: {train_features.shape}")
            print(f"   Testing set shape: {test_features.shape}")

            # Save processed features
            os.makedirs('data/processed', exist_ok=True)
            train_features.to_csv('data/processed/train_features.csv', index=False)
            test_features.to_csv('data/processed/test_features.csv', index=False)
            print("💾 Processed features saved to data/processed/")

            # Save checkpoint
            save_checkpoint('features_created', {
                'train_features': train_features,
                'test_features': test_features
            })
            
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            return
    else:
        print("\n⏭️  Skip Step 2: Feature Engineering (Using Checkpoint)")
        if os.path.exists('data/processed/train_features.csv') and os.path.exists('data/processed/test_features.csv'):
            print("📂 Loading feature data from files...")
            train_features = pd.read_csv('data/processed/train_features.csv')
            test_features = pd.read_csv('data/processed/test_features.csv')
            print(f"   Training set shape: {train_features.shape}")
            print(f"   Testing set shape: {test_features.shape}")
        else:
            checkpoint_data = load_checkpoint('features_created')
            if checkpoint_data is None:
                print("❌ Unable to load feature engineering checkpoint, please use --force-restart")
                return
            train_features = checkpoint_data['train_features']
            test_features = checkpoint_data['test_features']

    # 3. Model Training
    if start_step <= 3:
        print("\n🤖 Step 3: Model Training")
        print("-" * 30)

        trainer = ModelTrainer(n_folds=3, random_state=42)  # Reduce folds for faster demonstration
        
        try:
            # Prepare data
            X_train, y_train, X_test = trainer.prepare_data(train_features, test_features, 'TARGET')

            # Train LightGBM model
            print("\n🚀 Training LightGBM model...")
            lgb_result = trainer.train_lightgbm(X_train, y_train, X_test)

            print("✅ Model training completed!")
            print(f"   LightGBM CV AUC: {lgb_result['oof_auc']:.6f}")

            # Save checkpoint
            save_checkpoint('model_trained', {
                'lgb_result': lgb_result,
                'trainer': trainer
            })
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return
    else:
        print("\n⏭️  Skip Step 3: Model Training (Using Checkpoint)")
        checkpoint_data = load_checkpoint('model_trained')
        if checkpoint_data is None:
            print("❌ Unable to load model training checkpoint, please use --force-restart")
            return
        lgb_result = checkpoint_data['lgb_result']
        trainer = checkpoint_data['trainer']
        print(f"✅ Loaded trained model (CV AUC: {lgb_result['oof_auc']:.6f})")

    # 4. Generate Submission File
    if start_step <= 4:
        print("\n📝 Step 4: Generate Submission File")
        print("-" * 30)
        
        try:
            # Create submission directory
            os.makedirs('submissions', exist_ok=True)

            # Get test IDs and predictions
            test_ids = test_features['SK_ID_CURR'].values
            predictions = lgb_result['test_predictions']

            # Generate submission file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lgb_quickstart_{timestamp}.csv"
            
            trainer.create_submission(predictions, test_ids, filename)

            print("✅ Submission file generated successfully!")
            print(f"   File location: submissions/{filename}")
            print(f"   Number of predictions: {len(predictions)}")
            print(f"   Average prediction: {predictions.mean():.6f}")

            # Save checkpoint
            save_checkpoint('submission_created', {
                'filename': filename,
                'predictions': predictions,
                'test_ids': test_ids
            })
            
        except Exception as e:
            print(f"❌ Submission file generation failed: {e}")
            return
    else:
        print("\n⏭️  Skip Step 4: Generate Submission File (Completed)")

    # 5. Feature Importance Analysis
    if start_step <= 5:
        print("\n📈 Step 5: Feature Importance Analysis")
        print("-" * 30)
        
        try:
            feature_cols = [col for col in train_features.columns 
                           if col not in ['SK_ID_CURR', 'TARGET']]
            
            importance_df = trainer.get_feature_importance_df('lightgbm', feature_cols)
            
            if importance_df is not None:
                print("✅ Feature importance analysis completed!")
                print("\n🔝 Top 10 important features:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                    print(f"   {i:2d}. {row['feature']:<30} {row['importance']:.4f}")

                # Save feature importance
                importance_df.to_csv('data/processed/feature_importance.csv', index=False)
                print("\n💾 Feature importance saved to data/processed/feature_importance.csv")

        except Exception as e:
            print(f"⚠️  Feature importance analysis failed: {e}")
    else:
        print("\n⏭️  Skip Step 5: Feature Importance Analysis (Completed)")

    # Completion message
    print("\n🎉 Quick Start Completed!")

if __name__ == "__main__":
    main() 