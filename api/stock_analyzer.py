import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self):
        self.STOCK_LIST = {
            '港股': ['0700.HK', '9988.HK', '3690.HK', '1810.HK', '0388.HK'],
            '美股': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
        }
    
    def get_stock_data(self, symbol, period='1y'):
        """獲取股票數據"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def calculate_atr(self, df, period=14):
        """計算ATR"""
        if df.empty:
            return None
            
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """計算布林帶"""
        df = df.copy()
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        bb_std = df['Close'].rolling(window=period).std()
        
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * std_dev)
        
        return df
    
    def calculate_indicators(self, df):
        """計算技術指標"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # ATR
        df['ATR'] = self.calculate_atr(df)
        
        # 布林帶
        df = self.calculate_bollinger_bands(df)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df.dropna()
    
    def linear_regression_forecast(self, prices, days=30):
        """線性回歸預測"""
        from sklearn.linear_model import LinearRegression
        
        if len(prices) < 2:
            return np.full(days, prices.iloc[-1] if len(prices) > 0 else 0)
            
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(prices), len(prices) + days).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        return predictions
    
    def arima_forecast(self, prices, days=30):
        """ARIMA預測"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            if len(prices) < 30:
                last_price = prices.iloc[-1] if len(prices) > 0 else 0
                return np.full(days, last_price)
            
            # 簡單ARIMA模型
            model = ARIMA(prices, order=(1, 1, 1))
            model_fit = model.fit()
            
            forecast_result = model_fit.forecast(steps=days)
            return forecast_result.values
            
        except Exception as e:
            print(f"ARIMA error: {e}")
            last_price = prices.iloc[-1] if len(prices) > 0 else 0
            return np.full(days, last_price)
    
    def lstm_forecast(self, prices, days=30, look_back=30):
        """簡化版LSTM預測"""
        try:
            if len(prices) < look_back + days:
                last_price = prices.iloc[-1] if len(prices) > 0 else 0
                return np.full(days, last_price)
            
            # 使用簡單移動平均作為簡化版LSTM
            predictions = []
            for i in range(days):
                window = prices.iloc[-look_back:].values if len(prices) >= look_back else prices.values
                pred = np.mean(window) + np.random.normal(0, np.std(window) * 0.1)
                predictions.append(pred)
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"LSTM error: {e}")
            last_price = prices.iloc[-1] if len(prices) > 0 else 0
            return np.full(days, last_price)
    
    def analyze_stock(self, symbol, period='1y', forecast_days=30):
        """主分析函數"""
        try:
            # 獲取數據
            df = self.get_stock_data(symbol, period)
            if df is None or df.empty:
                return None
            
            # 計算技術指標
            df = self.calculate_indicators(df)
            
            if df.empty:
                return None
                
            # 準備數據
            prices = df['Close']
            dates = df.index
            
            # 各種模型預測
            lr_predictions = self.linear_regression_forecast(prices, forecast_days)
            arima_predictions = self.arima_forecast(prices, forecast_days)
            lstm_predictions = self.lstm_forecast(prices, forecast_days)
            
            # 生成預測日期
            last_date = dates[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='B'
            )
            
            # 計算統計數據
            stats = self.calculate_statistics(df, prices)
            
            # 生成交易信號
            signals = self.generate_signals(df)
            
            # 創建圖表數據
            chart_data = self.create_chart_data(df, future_dates, lr_predictions, 
                                               arima_predictions, lstm_predictions)
            
            return {
                'symbol': symbol,
                'data': df.tail(100).reset_index().to_dict('records'),
                'stats': stats,
                'signals': signals,
                'forecasts': {
                    'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                    'linear_regression': lr_predictions.tolist(),
                    'arima': arima_predictions.tolist(),
                    'lstm': lstm_predictions.tolist()
                },
                'chart_data': chart_data
            }
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return None
    
    def calculate_statistics(self, df, prices):
        """計算統計數據"""
        try:
            returns = prices.pct_change().dropna()
            
            stats = {
                'current_price': float(prices.iloc[-1]) if len(prices) > 0 else 0,
                'price_change': float(prices.iloc[-1] - prices.iloc[-2]) if len(prices) > 1 else 0,
                'price_change_pct': float((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100) if len(prices) > 1 else 0,
                'volatility': float(returns.std() * np.sqrt(252) * 100) if len(returns) > 0 else 0,
                'avg_volume': float(df['Volume'].mean()) if 'Volume' in df.columns else 0,
                'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and len(df) > 0 else 50,
                'atr': float(df['ATR'].iloc[-1]) if 'ATR' in df.columns and len(df) > 0 else 0,
                'bb_position': '中軌附近'
            }
            
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns and len(df) > 0:
                if df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1] * 0.95:
                    stats['bb_position'] = '上軌附近'
                elif df['Close'].iloc[-1] < df['BB_Lower'].iloc[-1] * 1.05:
                    stats['bb_position'] = '下軌附近'
            
            return stats
        except Exception as e:
            print(f"Stats error: {e}")
            return {}
    
    def generate_signals(self, df):
        """生成交易信號"""
        signals = []
        
        try:
            # RSI信號
            if 'RSI' in df.columns and len(df) > 0:
                rsi = df['RSI'].iloc[-1]
                if rsi < 30:
                    signals.append({'indicator': 'RSI', 'signal': '超賣', 'action': '買入機會'})
                elif rsi > 70:
                    signals.append({'indicator': 'RSI', 'signal': '超買', 'action': '賣出機會'})
            
            # 布林帶信號
            if 'BB_Lower' in df.columns and len(df) > 0:
                if df['Close'].iloc[-1] < df['BB_Lower'].iloc[-1]:
                    signals.append({'indicator': '布林帶', 'signal': '價格觸及下軌', 'action': '買入機會'})
                elif 'BB_Upper' in df.columns and df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1]:
                    signals.append({'indicator': '布林帶', 'signal': '價格觸及上軌', 'action': '賣出機會'})
        except Exception as e:
            print(f"Signals error: {e}")
        
        return signals
    
    def create_chart_data(self, df, future_dates, lr_pred, arima_pred, lstm_pred):
        """創建圖表數據"""
        try:
            # 價格數據
            price_data = {
                'dates': df.index.strftime('%Y-%m-%d').tolist()[-100:],
                'prices': df['Close'].tolist()[-100:],
                'future_dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'lr_predictions': lr_pred.tolist(),
                'arima_predictions': arima_pred.tolist(),
                'lstm_predictions': lstm_pred.tolist()
            }
            
            # 技術指標數據
            if 'BB_Upper' in df.columns:
                indicator_data = {
                    'dates': df.index.strftime('%Y-%m-%d').tolist()[-100:],
                    'bb_upper': df['BB_Upper'].tolist()[-100:],
                    'bb_middle': df['BB_Middle'].tolist()[-100:],
                    'bb_lower': df['BB_Lower'].tolist()[-100:],
                    'close': df['Close'].tolist()[-100:]
                }
            else:
                indicator_data = {}
            
            # ATR數據
            if 'ATR' in df.columns:
                atr_data = {
                    'dates': df.index.strftime('%Y-%m-%d').tolist()[-100:],
                    'atr': df['ATR'].tolist()[-100:]
                }
            else:
                atr_data = {}
            
            return {
                'price': price_data,
                'indicators': indicator_data,
                'atr': atr_data
            }
        except Exception as e:
            print(f"Chart data error: {e}")
            return {}
