import tensorflow as tf
import pandas as pd
LOSS_FUNCTION_NAME = 'sparse_categorical_crossentropy'

def basic_model(X, 
                Y, 
                epochs: int, 
                dropout: float = 0.2,
                optimizer: str = 'adam',
                ):
    
    input_size = X.shape[1]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=optimizer,
                  loss = LOSS_FUNCTION_NAME,
                  metrics=['accuracy'])
    
    model.fit(X, Y, epochs=epochs)

    return model
