import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Dropout
import streamlit as st

# Set page title and layout
st.set_page_config(page_title="Stock Price Prediction")

# Title and description
st.title("Stock Price Prediction Using LSTM")
st.write("""
This app predicts stock prices using a Long Short-Term Memory (LSTM) neural network.
Upload your dataset (CSV file) with columns: `Date`, `Open`, `Close`, and `Volume`.
""")

# Sidebar for user inputs
st.sidebar.header("User Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Function to load and preprocess data
def load_and_preprocess_data(file):
    df = pd.read_csv(file, parse_dates=['Date'])
    df.fillna(method='ffill', inplace=True)
    df = df[['Date', 'Open', 'Close', 'Volume']]
    return df

# Function to create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
        y.append(data[i, 1])  # Predict the 'Close' price
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main app logic
if uploaded_file is not None:
    # Load and preprocess data
    df = load_and_preprocess_data(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.write(df)

    # Normalize the data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Open', 'Close', 'Volume']])

    # Create sequences
    seq_length = 20
    X, y = create_sequences(df_scaled, seq_length)

    # Split the data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Extract dates for the test set
    test_dates = df['Date'].values[train_size + seq_length:]

    # Build and train the model
    st.subheader("Training the LSTM Model")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=7, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Plot training and validation loss
    st.write("Training and Validation Loss")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    st.pyplot(plt,use_container_width="True")

    # Make predictions
    y_pred = model.predict(X_test)

    # Inverse transform predictions and actual values
    y_pred_concat = np.concatenate((X_test[:, -1, 0:1], y_pred.reshape(-1, 1), X_test[:, -1, 2:3]), axis=1)
    y_pred_actual = scaler.inverse_transform(y_pred_concat)[:, 1]

    y_test_concat = np.concatenate((X_test[:, -1, 0:1], y_test.reshape(-1, 1), X_test[:, -1, 2:3]), axis=1)
    y_test_actual = scaler.inverse_transform(y_test_concat)[:, 1]

    # Plot actual vs predicted prices
    st.subheader("Actual vs Predicted Stock Prices")
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_actual, color='blue', label='Actual Stock Price')
    plt.plot(test_dates, y_pred_actual, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(plt,use_container_width="True")

    # Save the model as a .keras file
    model.save('stock_price_lstm_model.keras')
    st.success("Model saved as `stock_price_lstm_model.keras`")

else:
    st.info("Please upload a CSV file to get started.")