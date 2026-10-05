# train.py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data import load_raw_data, process_train_data, process_val_data
from model import build_model
import config
import os

def train():
    # 1. Ham verileri yükle
    train_df, _ = load_raw_data()
    
    # 2. Veriyi sızıntı olmaması için ÖNCE Train ve Validation olarak böl
    train_subset, val_subset = train_test_split(
        train_df, 
        test_size=config.VALIDATION_SPLIT, 
        random_state=42
    )

    # 3. Preprocessing (Imputer & Scaler) SADECE eğitim kümesinden öğrenilir!
    X_train, y_train = process_train_data(train_subset)
    # Validation kümesi sadece dönüştürülür (fit edilmez)
    X_val, y_val = process_val_data(val_subset)

    # 4. Modeli Yükle veya Kur
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"'{config.MODEL_SAVE_PATH}' yükleniyor. Eğitime KALINAN YERDEN devam edilecek...")
        model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    else:
        print("Kayıtlı model bulunamadı. SIFIRDAN yeni bir model kuruluyor...")
        model = build_model(input_dim=X_train.shape[1])

        if config.OPTIMIZER == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        elif config.OPTIMIZER == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE)
            
        model.compile(optimizer=optimizer, loss=config.LOSS, metrics=config.METRICS)
    
    # 5. Eğitimi Başlat (Artık validation_split yerine doğrudan X_val, y_val veriyoruz)
    print(f"{config.EPOCHS} epoch boyunca eğitim yapılıyor...")
    model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS, 
        batch_size=config.BATCH_SIZE,
        verbose=1
    )
    
    model.save(config.MODEL_SAVE_PATH)
    print(f"\nModel '{config.MODEL_SAVE_PATH}' dosyasına kaydedildi.")

if __name__ == "__main__":
    train()