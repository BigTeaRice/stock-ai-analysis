from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.io as pio
from models.lstm_model import LSTMModel
from models.arima_model import ARIMAModel
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# 支持的股票列表
STOCK_LIST = {
    '港股': ['0700.HK', '9988.HK', '3690.HK', '1810.HK', '0388.HK'],
    '美股': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
}

class StockAnalyzer:
    def __init__(self):
        self.lstm_model = LSTMModel()
        self.arima_model = ARIMAModel()
    
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
        """計算ATR (Average True Range)"""
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
        """計算布林帶 (Bollinger Bands)"""
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        bb_std = df['Close'].rolling(window=period).std()
        
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * std_dev)
        
        return df
    
    def calculate_indicators(self, df):
        """計算所有技術指標"""
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
        
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(prices), len(prices) + days).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        return predictions
    
    def analyze_stock(self, symbol, period='1y', forecast_days=30):
        """主分析函數"""
        # 獲取數據
        df = self.get_stock_data(symbol, period)
        if df is None or df.empty:
            return None
        
        # 計算技術指標
        df = self.calculate_indicators(df)
        
        # 準備數據
        prices = df['Close']
        dates = df.index
        
        # 線性回歸預測
        lr_predictions = self.linear_regression_forecast(prices, forecast_days)
        
        # ARIMA預測
        arima_predictions, arima_ci = self.arima_model.forecast(prices, forecast_days)
        
        # LSTM預測
        lstm_predictions = self.lstm_model.forecast(prices, forecast_days)
        
        # 生成預測日期
        last_date = dates[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='B'
        )
        
        # 創建圖表
        charts = self.create_charts(df, future_dates, lr_predictions, 
                                   arima_predictions, lstm_predictions)
        
        # 計算統計數據
        stats = self.calculate_statistics(df, prices)
        
        # 生成交易信號
        signals = self.generate_signals(df)
        
        return {
            'symbol': symbol,
            'data': df.tail(100).to_dict('records'),
            'charts': charts,
            'stats': stats,
            'signals': signals,
            'forecasts': {
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'linear_regression': lr_predictions.tolist(),
                'arima': arima_predictions.tolist(),
                'lstm': lstm_predictions.tolist(),
                'arima_ci': arima_ci.tolist() if arima_ci is not None else []
            }
        }
    
    def create_charts(self, df, future_dates, lr_pred, arima_pred, lstm_pred):
        """創建交互式圖表"""
        charts = {}
        
        # 價格走勢圖
        fig_price = go.Figure()
        
        # 實際價格
        fig_price.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='實際價格',
            line=dict(color='blue', width=2)
        ))
        
        # 預測價格
        fig_price.add_trace(go.Scatter(
            x=future_dates,
            y=lr_pred,
            mode='lines',
            name='線性回歸預測',
            line=dict(color='green', dash='dash')
        ))
        
        fig_price.add_trace(go.Scatter(
            x=future_dates,
            y=arima_pred,
            mode='lines',
            name='ARIMA預測',
            line=dict(color='red', dash='dash')
        ))
        
        fig_price.add_trace(go.Scatter(
            x=future_dates,
            y=lstm_pred,
            mode='lines',
            name='LSTM預測',
            line=dict(color='purple', dash='dash')
        ))
        
        fig_price.update_layout(
            title='股票價格及AI預測',
            xaxis_title='日期',
            yaxis_title='價格',
            hovermode='x unified'
        )
        
        charts['price'] = pio.to_html(fig_price, full_html=False)
        
        # 技術指標圖
        fig_indicators = go.Figure()
        
        # 布林帶
        fig_indicators.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            mode='lines',
            name='布林帶上軌',
            line=dict(color='gray', width=1)
        ))
        
        fig_indicators.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_Middle'],
            mode='lines',
            name='布林帶中軌',
            line=dict(color='black', width=1)
        ))
        
        fig_indicators.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            mode='lines',
            name='布林帶下軌',
            line=dict(color='gray', width=1),
            fill='tonexty'
        ))
        
        fig_indicators.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='收盤價',
            line=dict(color='blue', width=2)
        ))
        
        fig_indicators.update_layout(
            title='技術指標 - 布林帶',
            xaxis_title='日期',
            yaxis_title='價格'
        )
        
        charts['indicators'] = pio.to_html(fig_indicators, full_html=False)
        
        # ATR圖
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(
            x=df.index,
            y=df['ATR'],
            mode='lines',
            name='ATR',
            line=dict(color='orange', width=2)
        ))
        
        fig_atr.update_layout(
            title='平均真實波幅 (ATR)',
            xaxis_title='日期',
            yaxis_title='ATR值'
        )
        
        charts['atr'] = pio.to_html(fig_atr, full_html=False)
        
        return charts
    
    def calculate_statistics(self, df, prices):
        """計算統計數據"""
        returns = prices.pct_change().dropna()
        
        stats = {
            'current_price': float(prices.iloc[-1]),
            'price_change': float(prices.iloc[-1] - prices.iloc[-2]),
            'price_change_pct': float((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100),
            'volatility': float(returns.std() * np.sqrt(252) * 100),  # 年化波動率
            'avg_volume': float(df['Volume'].mean()),
            'rsi': float(df['RSI'].iloc[-1]),
            'atr': float(df['ATR'].iloc[-1]),
            'bb_position': '上軌附近' if df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1] * 0.95 
                          else '下軌附近' if df['Close'].iloc[-1] < df['BB_Lower'].iloc[-1] * 1.05
                          else '中軌附近'
        }
        
        return stats
    
    def generate_signals(self, df):
        """生成交易信號"""
        signals = []
        
        # RSI信號
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            signals.append({'indicator': 'RSI', 'signal': '超賣', 'action': '買入'})
        elif rsi > 70:
            signals.append({'indicator': 'RSI', 'signal': '超買', 'action': '賣出'})
        
        # 布林帶信號
        if df['Close'].iloc[-1] < df['BB_Lower'].iloc[-1]:
            signals.append({'indicator': '布林帶', 'signal': '價格觸及下軌', 'action': '買入機會'})
        elif df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1]:
            signals.append({'indicator': '布林帶', 'signal': '價格觸及上軌', 'action': '賣出機會'})
        
        # MACD信號
        if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['Signal'].iloc[-2]:
            signals.append({'indicator': 'MACD', 'signal': '金叉', 'action': '買入'})
        elif df['MACD'].iloc[-1] < df['Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['Signal'].iloc[-2]:
            signals.append({'indicator': 'MACD', 'signal': '死叉', 'action': '賣出'})
        
        return signals

analyzer = StockAnalyzer()

@app.route('/')
def index():
    return render_template('index.html', stock_list=STOCK_LIST)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        symbol = data.get('symbol')
        period = data.get('period', '1y')
        forecast_days = int(data.get('forecast_days', 30))
        
        result = analyzer.analyze_stock(symbol, period, forecast_days)
        
        if result:
            return jsonify({'success': True, 'data': result})
        else:
            return jsonify({'success': False, 'error': '無法獲取股票數據'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stocks')
def get_stocks():
    market = request.args.get('market', '港股')
    return jsonify(STOCK_LIST.get(market, []))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
