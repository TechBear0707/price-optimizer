import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
from copy import deepcopy
from datetime import date, timedelta
import datetime
from datetime import timedelta
from tensorflow.keras import backend as K
import os

# Set TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')


def lstm_forecast(df, product_dict):
    """
    Forecast sales for the next thirty days using LSTM.

    :param df: DataFrame containing sales data. Must have columns 'ProductID', 'TicketType', 'Quantity'.
    :param product_dict: Dictionary containing ProductID as key and list of TicketType as value.
    :return: The next thirty days forecasted sales for each product and ticket type.
    """
    next_thirty_days = pd.DataFrame(columns=['ProductID', 'TicketType', 'Quantity'])

    def clear_tuner_directory(directory='my_dir'):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    for product in product_dict.keys():
        prod_df = df.loc[df['ProductID'] == product]
        for ticket in product_dict[product]:
            ticket_df = prod_df.loc[prod_df['TicketType'] == ticket]

            # Ensure the index is a proper date range
            ticket_df.index = pd.to_datetime(ticket_df.index)
            full_date_range = pd.date_range(start=ticket_df.index.min(), end=ticket_df.index.max())
            ticket_df = ticket_df.reindex(full_date_range, fill_value=0)

            # Skip if data is insufficient
            sample = ticket_df.tail(30)
            if sum(sample['Quantity']) == 0:
                continue

            try:
                temp = ticket_df['Quantity']
                min_temp = temp.min()
                max_temp = temp.max()

                # Handle case where min and max are the same
                if max_temp == min_temp:
                    ticket_df['Quantity_Normalized'] = 0
                else:
                    ticket_df['Quantity_Normalized'] = (temp - min_temp) / (max_temp - min_temp)

                def df_to_X_y(df, window_size):
                    df_as_np = df.to_numpy()
                    X, y = [], []
                    for i in range(len(df_as_np) - window_size):
                        row = [[a] for a in df_as_np[i:i+window_size]]
                        X.append(row)
                        label = df_as_np[i+window_size]
                        y.append(label)
                    return np.array(X), np.array(y)

                WINDOW_SIZE = 30
                temp = ticket_df['Quantity_Normalized']
                X, y = df_to_X_y(temp, WINDOW_SIZE)

                # Validate data dimensions
                if X.shape[0] == 0 or y.shape[0] == 0:
                    print(f"Insufficient data for ProductID {product}, TicketType {ticket}.")
                    continue

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                def build_model(hp):
                    model = Sequential()
                    model.add(LSTM(hp.Int('n_neurons', 100, 200),
                                   activation='relu',
                                   input_shape=(WINDOW_SIZE, 1)))
                    model.add(Dense(hp.Int('second_layer', 50, 100),
                                    activation='relu'))
                    model.add(Dense(1, activation='linear'))
                    model.compile(optimizer=Adam(learning_rate=0.0001),
                                  loss=MeanSquaredError(),
                                  metrics=[RootMeanSquaredError()])
                    return model

                clear_tuner_directory()

                tuner = kt.BayesianOptimization(
                    build_model,
                    objective='val_loss',
                    max_trials=5,
                    directory='my_dir',
                    project_name='lstm_optimization'
                )

                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

                tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                model = tuner.hypermodel.build(best_hps)
                model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

                plt.plot(pd.DataFrame(model.history.history)['loss'], label='Loss')
                plt.plot(pd.DataFrame(model.history.history)['val_loss'], label='Validation Loss')
                plt.legend(loc=0)

                test_predictions = model.predict(X_test).flatten()

                test_pred_original = test_predictions * (max_temp - min_temp) + min_temp
                y_test_original = y_test * (max_temp - min_temp) + min_temp

                test_results = pd.DataFrame({'actual': y_test_original, 'predicted': test_pred_original})
                test_results.plot()

                future_date_range = pd.date_range(ticket_df.index[-1] + timedelta(days=1), periods=30)
                forecast = []

                Xin = X_test[-1:, :, :]

                def insert_end(Xin, new_input):
                    for i in range(WINDOW_SIZE-1):
                        Xin[:, i, :] = Xin[:, i+1, :]
                    Xin[:, WINDOW_SIZE-1, :] = new_input
                    return Xin

                for i in range(30):
                    out = model.predict(Xin, batch_size=1)
                    forecast.append(out[0, 0])
                    Xin = insert_end(Xin, out[0, 0])

                forecasted_output = np.array(forecast) * (max_temp - min_temp) + min_temp
                df_result = pd.DataFrame({'Date': future_date_range, 'Forecasted': forecasted_output})

                new_row = pd.DataFrame({'ProductID': [product],
                                        'TicketType': [ticket],
                                        'Quantity': [df_result['Forecasted'].sum()]})

                if df_result['Forecasted'].sum() > 0:
                    next_thirty_days = pd.concat([next_thirty_days, new_row], ignore_index=True)

                plt.plot(df_result['Date'], df_result['Forecasted'])
            except Exception as e:
                print(f"Error for ProductID {product}, TicketType {ticket}: {e}")
            finally:
                K.clear_session()

    return next_thirty_days
