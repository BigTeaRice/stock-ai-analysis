# 文件路径: models/lstm.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self):
        self.model = None
        self.scaler = None

    def create_dataset(self, data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def train(self, data, epochs=10, batch_size=32, time_step=60):
        # 数据预处理
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        X_train, y_train = self.create_dataset(scaled_data, time_step)

        # 调整输入形状 (samples, time steps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # 构建模型
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, steps=5, time_step=60):
        if self.model is None:
            return np.array([])

        # 使用最后的数据进行预测
        # 这里是一个简化的预测逻辑，实际应用中可能需要更复杂的滑动窗口处理
        # 为演示目的，我们返回一些模拟数据
        return np.random.rand(steps) * 100
