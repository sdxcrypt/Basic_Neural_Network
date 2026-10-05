# train.py
import tensorflow as tf
from data import load_training_data
from model import build_model
import config
import os

def train():
    X, y = load_training_data()
    
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"'{config.MODEL_SAVE_PATH}' yükleniyor. Eğitime KALINAN YERDEN devam edilecek...")
        model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    else:
        print("Kayıtlı model bulunamadı. SIFIRDAN yeni bir model kuruluyor...")
        model = build_model(input_dim=X.shape[1])

        # Derleme ayarlarını config'den alıyoruz
        if config.OPTIMIZER == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        elif config.OPTIMIZER == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE)
            
        model.compile(optimizer=optimizer, loss=config.LOSS, metrics=config.METRICS)
    
    print(f"{config.EPOCHS} epoch boyunca eğitim yapılıyor...")
    model.fit(
        X, y, 
        epochs=config.EPOCHS, 
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT, 
        verbose=1
    )
    
    model.save(config.MODEL_SAVE_PATH)
    print("Eğitim tamamlandı ve model kaydedildi.")

if __name__ == "__main__":
    train()