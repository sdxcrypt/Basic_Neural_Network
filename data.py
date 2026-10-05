# data.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import config

def load_raw_data():
    """Ham veriyi diskten yükler ve eksik değer stratejisini uygular."""
    train_df = pd.read_csv(config.TRAINING_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    return train_df, test_df

def process_train_data(train_df):
    """Eğitim verisinden imputer ve scaler öğrenir, veriyi dönüştürür ve kaydeder."""
    X = train_df[config.FEATURE_COLS].values
    y = train_df[config.TARGET_COL].values

    # 1. Eksik Veri Stratejisini Eğitime Uygula ve Kaydet
    if config.IMPUTATION_STRATEGY == 'zero':
        X = train_df[config.FEATURE_COLS].fillna(0).values
    else:
        imputer = SimpleImputer(strategy=config.IMPUTATION_STRATEGY)
        X = imputer.fit_transform(X)
        joblib.dump(imputer, config.IMPUTER_SAVE_PATH)

    # 2. Ölçeklendiriciyi Eğitime Uygula ve Kaydet
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, config.SCALER_SAVE_PATH)

    return X_scaled, y

def process_val_data(val_df):
    """Doğrulama (Validation) verisini, eğitimden öğrenilen imputer ve scaler ile dönüştürür (Sızıntı olmasın diye fit edilmez!)."""
    X_val = val_df[config.FEATURE_COLS].values

    if config.IMPUTATION_STRATEGY == 'zero':
        X_val = val_df[config.FEATURE_COLS].fillna(0).values
    else:
        imputer = joblib.load(config.IMPUTER_SAVE_PATH)
        X_val = imputer.transform(X_val)

    scaler = joblib.load(config.SCALER_SAVE_PATH)
    X_val_scaled = scaler.transform(X_val)

    return X_val_scaled, val_df[config.TARGET_COL].values

def load_test_data():
    """Test verisini yükler ve kaydedilmiş imputer/scaler ile dönüştürür."""
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    X_test = test_df[config.FEATURE_COLS].values

    if config.IMPUTATION_STRATEGY == 'zero':
        X_test = test_df[config.FEATURE_COLS].fillna(0).values
    else:
        imputer = joblib.load(config.IMPUTER_SAVE_PATH)
        X_test = imputer.transform(X_test)

    scaler = joblib.load(config.SCALER_SAVE_PATH)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, test_df[config.ID_COL]