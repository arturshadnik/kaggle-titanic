import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2
import pandas as pd
import matplotlib.pyplot as plt

class BaseNetwork:

    def __init__(self, optimizer, loss_function, input_size, metrics=["accuracy"]) -> None:
        self.input_size = input_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

    def train_model(self, X_train, Y_train, X_val, Y_val, epochs, minibatch_size, callbacks_str=None):
        callbacks = []
        self.history = self.model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=minibatch_size,
            validation_data=(X_val,Y_val),
            # callbacks=callbacks
        )
    
    def inference_model(self, X_new):
        return self.model.predict(X_new)
    
    def plot_results(self):
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        train_bin_acc = self.history.history['binary_accuracy']
        val_bin_acc = self.history.history['val_binary_accuracy']
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(train_bin_acc)
        plt.plot(val_bin_acc)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.show()
    
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

    def __init__(self, optimizer, loss_function, input_size, dropout=0.3, metrics=["accuracy"]) -> None:
        super().__init__(optimizer, loss_function, input_size, metrics)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
            tf.keras.layers.Dropout(dropout),   
            # tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
            # tf.keras.layers.Dropout(dropout),         
            tf.keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=l2(0.001)),            
        ])

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )