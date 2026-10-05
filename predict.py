# predict.py
import tensorflow as tf
import pandas as pd
from data import load_test_data
import config

def predict():
    print("Test verileri (Imputer ve Scaler ile) yükleniyor...")
    X_test, test_ids = load_test_data()

    print("Eğitilmiş model yükleniyor...")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    print("Test verisi üzerinden tahminler yapılıyor...")
    tahminler = model.predict(X_test)

    print("Kaggle submission dosyası hazırlanıyor...")
    submission_df = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    if len(tahminler) == len(submission_df):
        # Şablondaki hedef kolonun ismini otomatik bul (Örn: 'mu', 'power' veya 'fiyat')
        hedef_kolon_adi = submission_df.columns[1] 
        
        # Tahminleri yerleştir (Eğer çıktımız 1 nöron ise)
        submission_df[hedef_kolon_adi] = tahminler[:, 0]

        # Fiziksel olarak eksi değerde enerji üretilmez, negatifleri 0 yap
        submission_df[hedef_kolon_adi] = submission_df[hedef_kolon_adi].clip(lower=0)

        submission_df.to_csv(config.PREDICTION_SAVE_PATH, index=False)
        print(f"\nBAŞARILI: Tahmin dosyası '{config.PREDICTION_SAVE_PATH}' konumunda oluşturuldu.")
    else:
        print(f"KRİTİK HATA: Tahmin sayısı ({len(tahminler)}) ile Kaggle şablon satır sayısı ({len(submission_df)}) uyuşmuyor!")

if __name__ == "__main__":
    predict()