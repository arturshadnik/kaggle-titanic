import tensorflow as tf
import pandas as pd

class BaseNetwork:

    def __init__(self, optimizer, loss_function, input_size, metrics=["accuracy"]) -> None:
        self.input_size = input_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

    def train_model(self, X_train, Y_train, X_val, Y_val, epochs, minibatch_size):
        self.model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=minibatch_size,
            validation_data=(X_val,Y_val)
        )
    
    def inference_model(self, X_new):
        return self.model.predict(X_new)
    
class LogisticRegression(BaseNetwork):

    def __init__(self, optimizer, loss_function, input_size, metrics=["accuracy"]) -> None:
        super().__init__(optimizer, loss_function, input_size, metrics)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.input_size),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

class DNN(BaseNetwork):

    def __init__(self, optimizer, loss_function, input_size, dropout=0.2, metrics=["accuracy"]) -> None:
        super().__init__(optimizer, loss_function, input_size, metrics)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )