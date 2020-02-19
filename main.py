import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def read_data(training_file, predictions_features_file):
    training = pd.read_csv(training_file)
    predictions = pd.read_csv(predictions_features_file)

    return training, predictions


def data_cleaning(training, predictions):
    # Identify training features
    categorical_cols = list(training.select_dtypes('object').columns)
    id_cols = ['encounter_id', 'patient_id', 'hospital_id']
    target_col = ['hospital_death']
    exclude_cols = categorical_cols + id_cols + target_col
    numerical_col = [col for col in training.columns if col not in exclude_cols]

    training_target = training[target_col]
    training_features = training[numerical_col]
    prediction_features = predictions[numerical_col]

    # handle missing data
    training_features.fillna(0, inplace=True)
    prediction_features.fillna(0, inplace=True)

    # scale data, note this turns DataFrames to numpy arrays
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(training_features)
    training_features = scaler.transform(training_features)
    prediction_features = scaler.transform(prediction_features)

    return training_target, training_features, prediction_features


if __name__ == '__main__':
    training_file = 'data/training_v2.csv'
    predictions_features_file = 'data/unlabeled.csv'

    training, predictions = read_data(training_file, predictions_features_file)
    training_target, training_features, prediction_features = data_cleaning(training, predictions)

    # Splits training and testing data, for measuring accuracy later on.
    X_train, X_test, y_train, y_test = train_test_split(training_features, training_target.values, test_size=0.1)

    # The neural network being created.
    model = keras.Sequential([keras.layers.Dense(units=174, activation='tanh', input_shape=[174]),
                              keras.layers.Dense(600, activation='relu'),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(600, activation='relu'),
                              keras.layers.Dense(1, activation='tanh')])

    # Compile and train the model based on the gathered training data.
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5)

    # Predict the y value based on the set of X testing values that were set aside.
    predictions = model.predict(X_test)

    # Get the accuracy from the real y values and the rounded predictions.
    # We have to round the predictions, because we cannot get the accuracy
    # of two targets with binary and continuous targets.
    accuracy = accuracy_score(y_test, predictions.round())

    print("Accuracy: ", accuracy)

    prediction_file = pd.read_csv(predictions_features_file)

    # Getting the predictions from the already set aside data
    real_predictions = model.predict(prediction_features)

    result = pd.DataFrame()
    result['encounter_id'] = prediction_file['encounter_id']
    result['hospital_death'] = real_predictions
    result.to_csv('data_predictions.csv', index=False)

