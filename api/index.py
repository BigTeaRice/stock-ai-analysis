from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from api.stock_analyzer import StockAnalyzer

app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'),
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))

analyzer = StockAnalyzer()

@app.route('/')
def index():
    return render_template('index.html', stock_list=analyzer.STOCK_LIST)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        symbol = data.get('symbol', '0700.HK')
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
    return jsonify(analyzer.STOCK_LIST.get(market, []))

# Vercel 需要這個變量
app = app
