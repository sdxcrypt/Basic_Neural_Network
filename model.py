# model.py
import tensorflow as tf
import config

def build_model(input_dim):
    model = tf.keras.Sequential()
    
    # 1. İlk Katman
    model.add(tf.keras.layers.Dense(
        config.HIDDEN_LAYERS[0], 
        input_dim=input_dim, 
        activation=config.HIDDEN_ACTIVATION
    ))
    
    if config.DROPOUT_RATE > 0:
        model.add(tf.keras.layers.Dropout(config.DROPOUT_RATE))
        
    # 2. Ara Katmanlar
    for units in config.HIDDEN_LAYERS[1:]:
        model.add(tf.keras.layers.Dense(units, activation=config.HIDDEN_ACTIVATION))
        
    # 3. Çıktı Katmanı
    model.add(tf.keras.layers.Dense(
        config.OUTPUT_NEURONS, 
        activation=config.OUTPUT_ACTIVATION
    ))
    
    return model