"""
股票分析模型套件

這個套件包含以下模型：
1. LSTMModel - 基於深度學習的時間序列預測
2. ARIMAModel - 傳統時間序列分析
"""

import warnings
warnings.filterwarnings('ignore')

# 導入子模塊
from .lstm_model import LSTMModel
from .arima_model import ARIMAModel
from .linear_regression import LinearRegressionModel

# 套件元數據
__version__ = '1.0.0'
__author__ = 'AI Stock Analysis Team'
__email__ = 'support@stock-analysis.com'

# 控制導入行為
__all__ = [
    'LSTMModel',
    'ARIMAModel', 
    'LinearRegressionModel',
    'create_model',
    'MODEL_TYPES'
]

# 套件級別的變量
MODEL_TYPES = {
    'lstm': '長短期記憶網絡',
    'arima': '自回歸整合移動平均',
    'linear': '線性回歸'
}

# 套件級別的函數
def create_model(model_type='lstm'):
    """
    創建指定類型的模型
    
    Args:
        model_type (str): 模型類型，可選 'lstm', 'arima', 'linear'
    
    Returns:
        模型實例
    """
    model_map = {
        'lstm': LSTMModel,
        'arima': ARIMAModel,
        'linear': LinearRegressionModel
    }
    
    if model_type not in model_map:
        raise ValueError(f"不支持的模型類型: {model_type}")
    
    return model_map[model_type]()

def get_model_info(model_type):
    """
    獲取模型信息
    
    Args:
        model_type (str): 模型類型
    
    Returns:
        dict: 模型信息
    """
    info_map = {
        'lstm': {
            'name': 'LSTM',
            'description': '長短期記憶網絡，適合時間序列預測',
            'parameters': ['look_back', 'units', 'dropout_rate']
        },
        'arima': {
            'name': 'ARIMA',
            'description': '自回歸整合移動平均模型',
            'parameters': ['p', 'd', 'q']
        }
    }
    
    return info_map.get(model_type, {})

# 初始化代碼
print(f"股票分析模型套件 v{__version__} 已加載")
