"""
API 套件 - 股票分析 REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# 添加父目錄到路徑，以便導入 models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .stock_analyzer import StockAnalyzer

class StockAPI:
    """
    股票分析 API 類
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # 啟用跨域請求
        self.analyzer = StockAnalyzer()
        self._setup_routes()
    
    def _setup_routes(self):
        """設置 API 路由"""
        
        @self.app.route('/')
        def index():
            """首頁"""
            return jsonify({
                'status': 'online',
                'name': 'AI Stock Analysis API',
                'version': '1.0.0'
            })
        
        @self.app.route('/api/health')
        def health_check():
            """健康檢查"""
            return jsonify({'status': 'healthy'})
        
        @self.app.route('/api/analyze', methods=['POST'])
        def analyze():
            """分析股票"""
            try:
                data = request.json
                symbol = data.get('symbol', '0700.HK')
                period = data.get('period', '1y')
                forecast_days = int(data.get('forecast_days', 30))
                
                result = self.analyzer.analyze_stock(symbol, period, forecast_days)
                
                if result:
                    return jsonify({
                        'success': True,
                        'data': result
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': '無法獲取股票數據'
                    }), 400
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/stocks')
        def get_stocks():
            """獲取支持的股票列表"""
            market = request.args.get('market', '港股')
            stocks = self.analyzer.STOCK_LIST.get(market, [])
            return jsonify(stocks)
        
        @self.app.route('/api/models')
        def get_models():
            """獲取支持的模型列表"""
            return jsonify({
                'models': ['LSTM', 'ARIMA', 'Linear Regression'],
                'indicators': ['ATR', 'Bollinger Bands', 'RSI', 'MACD']
            })
    
    def run(self, debug=False, host='0.0.0.0', port=5000):
        """運行 API 服務器"""
        self.app.run(debug=debug, host=host, port=port)

# 創建 API 實例
api_app = StockAPI().app

# 方便導入的別名
app = api_app

# 套件級別的函數
def create_api():
    """創建 API 實例"""
    return StockAPI()

def get_stock_list(market='港股'):
    """獲取股票列表"""
    analyzer = StockAnalyzer()
    return analyzer.STOCK_LIST.get(market, [])
