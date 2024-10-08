# config.py

import os

# File paths
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUBMISSION_PATH = 'data/submission.csv'
README_PATH = 'data/README.txt'

# Paths to saved objects
MODELS_DIR = 'models'
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
PCA_PATH = os.path.join(MODELS_DIR, 'pca.pkl')
ENCODERS_PATH = os.path.join(MODELS_DIR, 'encoders.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'mlp2.pth')

# Columns for encoding
ONE_HOT_COLUMNS = ['Sex', 'Resting_electrocardiographic_results']
LABEL_ENCODE_COLUMNS = ['Number_of_major_vessels', 'Thal', 'Chest_bin', 'Slope']

# PCA settings
PCA_COMPONENTS = 5

# Neural Network settings
MLP2_HIDDEN_LAYERS = [64, 128, 64, 32]
MLP_LEARNING_RATE = 0.001
MLP_EPOCHS = 100
MLP_BATCH_SIZE = 32

# Numeric columns for boxplot and outlier removal
NUMERIC_COLUMNS = [
    'Age',
    'Resting_blood_pressure',
    'Serum_cholestoral',
    'Fasting_blood_sugar',
    'Maximum_heart_rate_achieved',
    'Oldpeak'
]
OUTLIER_COLUMNS = [
    'Resting_blood_pressure',
    'Serum_cholestoral',
    'Fasting_blood_sugar',
    'Maximum_heart_rate_achieved',
    'Oldpeak'
]

# Target column
TARGET_COLUMN = 'class'

# Scatter plot settings
SCATTER_PLOT_COLUMNS = ['Age', 'Maximum_heart_rate_achieved']
