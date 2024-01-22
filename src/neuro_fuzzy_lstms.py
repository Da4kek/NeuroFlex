import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback


class NeuroGenesis:
    def __init__(self, architecture, max_neurons, threshold_epochs):
        self.architecture = architecture
        self.max_neurons = max_neurons
        self.threshold_epochs = threshold_epochs
        self.consecutive_epochs_no_improvement = 0

    def trigger(self, current_epoch, current_accuracy, previous_accuracy):
        if current_accuracy <= previous_accuracy:
            self.consecutive_epochs_no_improvement += 1
        else:
            self.consecutive_epochs_no_improvement = 0
        if self.consecutive_epochs_no_improvement >= self.threshold_epochs:
            self.consecutive_epochs_no_improvement = 0
            self.neurogenesis()

    def neurogenesis(self):
        if self.architecture.num_neurons < self.max_neurons:
            new_lstm_cell = LSTM(self.architecture.num_neurons, input_shape=(
                1, self.architecture.input_size), return_sequences=True)
            self.architecture.model.add(new_lstm_cell)
            self.architecture.num_neurons += 1


class NeuroFuzzyNetwork:
    def __init__(self, input_size, output_size, num_rules, num_neurons, max_neurons, threshold_epochs):
        self.input_size = input_size
        self.output_size = output_size
        self.num_rules = num_rules
        self.num_neurons = num_neurons
        self.model = self.build_model()
        self.previous_accuracy = None
        self.max_neurons = max_neurons
        self.threshold_epochs = threshold_epochs

    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.num_neurons, input_shape=(1, self.input_size)))
        model.add(Dense(self.num_rules, activation='sigmoid'))
        model.add(Dense(self.output_size, activation='softmax'))
        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        history_list = []  
        for epoch in range(epochs):
            history = self.model.fit(
                X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
            history_list.append(history.history)
            current_accuracy = history.history['accuracy'][0]

            if epoch > 0 and self.previous_accuracy is not None:
                previous_accuracy = self.previous_accuracy
                NeuroGenesis(self, self.max_neurons, self.threshold_epochs).trigger(
                    epoch, current_accuracy, previous_accuracy)

            self.previous_accuracy = current_accuracy

        return history_list

    def predict(self, X_test):
        return self.model.predict(X_test)



class PreviousAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.previous_accuracy = logs.get('accuracy')
