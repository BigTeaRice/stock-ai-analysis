import os
import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg') # 防止在无GUI环境下报错
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# 获取股票数据
def get_stock_data(ticker):
    """使用yfinance获取数据"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo") # 获取6个月数据
        if hist.empty:
            return None
        return hist
    except:
        return None

# 计算Alpha因子和指标
def calculate_factors(df):
    """计算多个量化指标"""
    # 1. 动量因子 (20日收益率)
    df['Momentum'] = df['Close'] / df['Close'].shift(20) - 1

    # 2. 波动率因子 (20日标准差)
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()

    # 3. 布林带
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)

    # 4. 简单线性回归预测 (用于绘图)
    try:
        df['Time'] = np.arange(len(df))
        X = df[['Time']].dropna()
        y = df['Close'].dropna()
        model = LinearRegression().fit(X, y)
        df['Prediction'] = model.predict(df[['Time']])
    except:
        df['Prediction'] = np.nan

    return df

# 生成图表
def generate_plot(df, ticker):
    """生成K线图和指标图"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.plot(df.index, df['MA20'], label='20-day MA', color='orange', alpha=0.7)
    plt.plot(df.index, df['Upper_Band'], label='Upper Band', color='red', linestyle='--', alpha=0.5)
    plt.plot(df.index, df['Lower_Band'], label='Lower Band', color='green', linestyle='--', alpha=0.5)
    plt.title(f'Stock Analysis: {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # 将图表转换为Base64字符串
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# Flask路由
@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    stock_data = None
    plot_url = None
    factors = None

    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip().upper()

        # 简单验证代码格式
        if not ticker.endswith('.HK'):
            ticker += '.HK'

        df = get_stock_data(ticker)
        if df is None:
            error = f"无法获取股票 {ticker} 的数据，请检查代码或网络连接。"
        else:
            # 计算因子
            df = calculate_factors(df)

            # 生成图表
            plot_url = generate_plot(df, ticker)

            # 准备展示的数据 (取最后几行)
            stock_data = df[['Close', 'Volume', 'Momentum', 'Volatility']].tail(5).to_html()

            # 准备因子摘要
            latest = df.iloc[-1]
            factors = {
                '最新价格': f"{latest['Close']:.2f}",
                '动量 (20日)': f"{latest['Momentum']:.2%}" if not pd.isna(latest['Momentum']) else "N/A",
                '波动率 (20日)': f"{latest['Volatility']:.2%}" if not pd.isna(latest['Volatility']) else "N/A",
                '布林带位置': f"上轨: {latest['Upper_Band']:.2f} | 中轨: {latest['MA20']:.2f} | 下轨: {latest['Lower_Band']:.2f}"
            }

    # 内联HTML模板 (为了方便你复制，直接写在代码里)
    html_template = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>本地量化股票分析</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f4; }
            .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
            input[type="text"] { width: 70%; padding: 10px; margin-right: 10px; border: 1px solid #ccc; border-radius: 4px; }
            input[type="submit"] { padding: 10px 20px; background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; }
            input[type="submit"]:hover { background: #0056b3; }
            .error { color: red; margin-top: 10px; }
            .result { margin-top: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            table, th, td { border: 1px solid #ddd; }
            th, td { padding: 12px; text-align: left; }
            .factor-item { margin-bottom: 5px; }
            img { max-width: 100%; height: auto; margin-top: 20px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>本地量化股票分析</h1>
            <form method="POST">
                <input type="text" name="ticker" placeholder="输入股票代码，例如 0700.HK" value="{{ request.form.ticker or '' }}">
                <input type="submit" value="分析">
            </form>

            {% if error %}
                <div class="error">{{ error }}</div>
            {% endif %}

            {% if plot_url %}
                <div class="result">
                    <h3>图表分析</h3>
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Stock Chart">
                </div>
            {% endif %}

            {% if factors %}
                <div class="result">
                    <h3>量化因子摘要</h3>
                    <div class="factor-item"><strong>最新价格:</strong> {{ factors['最新价格'] }}</div>
                    <div class="factor-item"><strong>动量 (20日):</strong> {{ factors['动量 (20日)'] }}</div>
                    <div class="factor-item"><strong>波动率 (20日):</strong> {{ factors['波动率 (20日)'] }}</div>
                    <div class="factor-item"><strong>布林带:</strong> {{ factors['布林带位置'] }}</div>
                </div>
            {% endif %}

            {% if stock_data %}
                <div class="result">
                    <h3>近期数据 (最后5条)</h3>
                    {{ stock_data|safe }}
                </div>
            {% endif %}
        </div>
    </body>
    </html>
    '''

    return render_template_string(html_template, error=error, plot_url=plot_url, factors=factors, stock_data=stock_data)

if __name__ == '__main__':
    # 为了在GitHub Pages上运行，需要监听0.0.0.0和端口5000
    app.run(host='0.0.0.0', port=5000, debug=True)