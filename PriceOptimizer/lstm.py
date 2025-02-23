import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split
from datetime import timedelta
from keras import backend as K

# Set TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')


def lstm_forecast(df, product_dict=None, aggregate_by_product=True):
    next_thirty_days = pd.DataFrame(columns=['ProductID', 'TicketType', 'Quantity'])

    def clear_tuner_directory(directory='my_dir'):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    df['StayDate'] = pd.to_datetime(df['StayDate'])

    if aggregate_by_product:
        df = df.groupby(['ProductID', 'StayDate']).agg({'Quantity': 'sum'}).reset_index()
        product_keys = df['ProductID'].unique()
    else:
        product_keys = product_dict.keys()

    for product in product_keys:
        prod_df = df[df['ProductID'] == product]

        if aggregate_by_product:
            ticket_types = [None]  # Dummy ticket type since we are aggregating at product level
        else:
            ticket_types = product_dict.get(product, [])

        for ticket in ticket_types:
            if not aggregate_by_product:
                ticket_df = prod_df[prod_df['TicketType'] == ticket]
            else:
                ticket_df = prod_df.copy()
                ticket_df['TicketType'] = None  # Dummy ticket type

            ticket_df.set_index('StayDate', inplace=True)
            full_date_range = pd.date_range(start=ticket_df.index.min(), end=ticket_df.index.max())
            ticket_df = ticket_df.reindex(full_date_range, fill_value=0)

            sample = ticket_df.tail(30)
            if sum(sample['Quantity']) == 0:
                continue

            try:
                temp = ticket_df['Quantity']
                min_temp, max_temp = temp.min(), temp.max()

                if max_temp == min_temp:
                    ticket_df['Quantity_Normalized'] = 0
                else:
                    ticket_df['Quantity_Normalized'] = (temp - min_temp) / (max_temp - min_temp)

                def df_to_X_y(df, window_size):
                    df_as_np = df.to_numpy()
                    X, y = [], []
                    for i in range(len(df_as_np) - window_size):
                        row = [[a] for a in df_as_np[i:i + window_size]]
                        X.append(row)
                        y.append(df_as_np[i + window_size])
                    return np.array(X), np.array(y)

                WINDOW_SIZE = 30
                temp = ticket_df['Quantity_Normalized']
                X, y = df_to_X_y(temp, WINDOW_SIZE)

                if X.shape[0] == 0 or y.shape[0] == 0:
                    print(f"Insufficient data for ProductID {product}, TicketType {ticket}.")
                    continue

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                def build_model(hp):
                    model = Sequential()
                    model.add(LSTM(hp.Int('n_neurons', 100, 200), activation='relu', input_shape=(WINDOW_SIZE, 1)))
                    model.add(Dense(hp.Int('second_layer', 50, 100), activation='relu'))
                    model.add(Dense(1, activation='linear'))
                    model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError(),
                                  metrics=[RootMeanSquaredError()])
                    return model

                clear_tuner_directory()
                tuner = kt.BayesianOptimization(build_model, objective='val_loss', max_trials=5,
                                                directory='my_dir', project_name='lstm_optimization')
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

                tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                model = tuner.hypermodel.build(best_hps)
                model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

                future_date_range = pd.date_range(ticket_df.index[-1] + timedelta(days=1), periods=30)
                forecast = []
                Xin = X_test[-1:, :, :]

                def insert_end(Xin, new_input):
                    for i in range(WINDOW_SIZE - 1):
                        Xin[:, i, :] = Xin[:, i + 1, :]
                    Xin[:, WINDOW_SIZE - 1, :] = new_input
                    return Xin

                for i in range(30):
                    out = model.predict(Xin, batch_size=1)
                    forecast.append(out[0, 0])
                    Xin = insert_end(Xin, out[0, 0])

                forecasted_output = np.array(forecast) * (max_temp - min_temp) + min_temp
                df_result = pd.DataFrame({'Date': future_date_range, 'Forecasted': forecasted_output})

                new_row = pd.DataFrame({'ProductID': [product],
                                        'TicketType': [ticket if not aggregate_by_product else None],
                                        'Quantity': [df_result['Forecasted'].sum()]})

                if df_result['Forecasted'].sum() > 0:
                    next_thirty_days = pd.concat([next_thirty_days, new_row], ignore_index=True)

            except Exception as e:
                print(f"Error for ProductID {product}, TicketType {ticket}: {e}")
            finally:
                K.clear_session()

    return next_thirty_days
