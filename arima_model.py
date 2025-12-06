import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    def __init__(self):
        self.model = None
        
    def find_best_arima(self, data, max_p=5, max_d=2, max_q=5):
        """尋找最佳ARIMA參數"""
        best_aic = np.inf
        best_order = None
        
        for p in range(max_p):
            for d in range(max_d + 1):
                for q in range(max_q):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        results = model.fit(method_kwargs={'maxiter': 1000}, disp=0)
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def forecast(self, data, forecast_days=30):
        """使用ARIMA進行預測"""
        try:
            # 尋找最佳參數
            best_order = self.find_best_arima(data)
            
            if best_order is None:
                # 如果無法找到合適參數，使用簡單移動平均
                last_price = data.iloc[-1]
                return np.full(forecast_days, last_price), None
            
            # 訓練模型
            self.model = ARIMA(data, order=best_order)
            self.model_fit = self.model.fit()
            
            # 進行預測
            forecast_result = self.model_fit.get_forecast(steps=forecast_days)
            predictions = forecast_result.predicted_mean
            
            # 獲取置信區間
            conf_int = forecast_result.conf_int()
            
            return predictions.values, conf_int.values
            
        except Exception as e:
            print(f"ARIMA forecast error: {e}")
            # 返回簡單預測
            last_price = data.iloc[-1]
            return np.full(forecast_days, last_price), None
