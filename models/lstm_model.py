import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def prepare_data(self, data, look_back=60):
        """準備LSTM訓練數據"""
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, look_back=60):
        """構建LSTM模型"""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def forecast(self, data, forecast_days=30, look_back=60):
        """使用LSTM進行預測"""
        if len(data) < look_back + forecast_days:
            # 如果數據不足，使用簡單預測
            last_price = data.iloc[-1]
            return np.full(forecast_days, last_price)
        
        # 準備數據
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # 構建並訓練模型
        self.model = self.build_model(look_back)
        
        X, y = self.prepare_data(data, look_back)
        
        # 訓練模型
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # 進行預測
        predictions = []
        current_batch = scaled_data[-look_back:].reshape(1, look_back, 1)
        
        for _ in range(forecast_days):
            current_pred = self.model.predict(current_batch, verbose=0)
            predictions.append(current_pred[0, 0])
            current_batch = np.append(current_batch[:, 1:, :], 
                                     [[current_pred[0]]], 
                                     axis=1)
        
        # 反標準化
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        return predictions
