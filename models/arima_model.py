# 文件路径: models/arima.py
import warnings
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None

    def train(self, train_data):
        warnings.filterwarnings("ignore")
        # 确保输入是一维数组
        train_series = train_data.flatten()
        self.model = ARIMA(train_series, order=self.order).fit()

    def predict(self, steps=5):
        forecast = self.model.forecast(steps=steps)
        return forecast.values
