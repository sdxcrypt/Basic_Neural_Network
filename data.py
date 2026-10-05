# data.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import config

def load_training_data():
    train_df = pd.read_csv(config.TRAINING_DATA_PATH)
    
    X = train_df[config.FEATURE_COLS].values
    y = train_df[config.TARGET_COL].values

    # 1. Eksik Verileri Doldur (Imputation)
    if config.IMPUTATION_STRATEGY == 'zero':
        X = train_df[config.FEATURE_COLS].fillna(0).values
    else:
        imputer = SimpleImputer(strategy=config.IMPUTATION_STRATEGY)
        X = imputer.fit_transform(X)
        joblib.dump(imputer, config.IMPUTER_SAVE_PATH) # Test verisinde kullanmak üzere kaydet

    # 2. Ölçeklendirme (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, config.SCALER_SAVE_PATH)

    return X_scaled, y

def load_test_data():
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    X_test = test_df[config.FEATURE_COLS].values

    # 1. Eğitimde öğrenilen eksik veri stratejisini (Imputer) yükle ve uygula
    if config.IMPUTATION_STRATEGY == 'zero':
        X_test = test_df[config.FEATURE_COLS].fillna(0).values
    else:
        imputer = joblib.load(config.IMPUTER_SAVE_PATH)
        X_test = imputer.transform(X_test)

    # 2. Eğitimde öğrenilen ölçeklendiriciyi (Scaler) yükle ve uygula
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, test_df[config.ID_COL]