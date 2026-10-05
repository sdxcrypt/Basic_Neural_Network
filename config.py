# config.py

# ==========================================
# 1. DOSYA YOLLARI (PATHS)
# ==========================================
TRAINING_DATA_PATH = "data/raw/train.csv"
TEST_DATA_PATH = "data/raw/test.csv"
SAMPLE_SUBMISSION_PATH = "data/raw/sample_submission.csv"

MODEL_SAVE_PATH = "data/output/model.keras" 
SCALER_SAVE_PATH = "data/output/scaler.save"
IMPUTER_SAVE_PATH = "data/output/imputer.save"  # Eksik veri doldurucu modeli kaydetmek için
PREDICTION_SAVE_PATH = "data/output/prediction.csv"

# ==========================================
# 2. VERİ PARAMETRELERİ (DATA PARAMS)
# ==========================================
TARGET_COL = 'power' 
FEATURE_COLS = ['farm', 'horizon', 'u', 'v', 'ws', 'weather_available']
ID_COL = 'id'

IMPUTATION_STRATEGY = 'mean'
# 'zero' : Eksik yerlere direkt 0 basar (Çoğu zaman kötüdür)
# 'mean' : Sütunun ortalamasıyla doldurur (Standart yaklaşım)
# 'median': Sütunun ortanca değeriyle doldurur (Aykırı/uç değerler çoksa idealdir)
# 'most_frequent': En çok tekrar eden değerle doldurur

# ==========================================
# 3. MODEL MİMARİSİ (MODEL ARCHITECTURE)
# ==========================================
HIDDEN_LAYERS = [128, 64, 32]  
DROPOUT_RATE = 0.2             

HIDDEN_ACTIVATION = 'relu'
# 'relu' : Görüntü, regresyon ve klasik problemlerde en iyisi.
# 'tanh' : Çıktının -1 ile 1 arasında dalgalandığı durumlarda iyidir.
# 'elu' veya 'selu' : ReLU'nun ölü nöron problemini çözen gelişmiş alternatifleri.

OUTPUT_NEURONS = 1             
OUTPUT_ACTIVATION = 'linear'   

# ==========================================
# 4. EĞİTİM PARAMETRELERİ (TRAINING PARAMS)
# ==========================================
EPOCHS = 100
BATCH_SIZE = 64                
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2         

LOSS = 'mse' 
OPTIMIZER = 'adam'
METRICS = ['mae']