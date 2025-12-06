let currentAnalysis = null;

async function analyzeStock() {
    const symbol = document.getElementById('stockSelect').value;
    const period = document.getElementById('periodSelect').value;
    const forecastDays = document.getElementById('forecastDays').value;
    
    // 顯示加載動畫
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    // 禁用分析按鈕
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 分析中...';
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                period: period,
                forecast_days: forecastDays
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentAnalysis = result.data;
            displayResults(result.data);
        } else {
            alert('分析失敗: ' + result.error);
        }
    } catch (error) {
        alert('網絡錯誤: ' + error.message);
    } finally {
        // 隱藏加載動畫
        document.getElementById('loading').style.display = 'none';
        document.getElementById('results').style.display = 'block';
        
        // 恢復分析按鈕
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> 開始分析';
    }
}

function displayResults(data) {
    displayStockInfo(data);
    displaySignals(data.signals);
    displayCharts(data.charts);
    displayForecastTable(data.forecasts);
}

function displayStockInfo(data) {
    const stats = data.stats;
    const infoHtml = `
        <div class="col-md-3">
            <div class="stock-info-card">
                <h6>當前價格</h6>
                <div class="value">$${stats.current_price.toFixed(2)}</div>
                <div class="${stats.price_change >= 0 ? 'change-positive' : 'change-negative'}">
                    ${stats.price_change >= 0 ? '+' : ''}${stats.price_change.toFixed(2)} 
                    (${stats.price_change >= 0 ? '+' : ''}${stats.price_change_pct.toFixed(2)}%)
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stock-info-card">
                <h6>波動率</h6>
                <div class="value">${stats.volatility.toFixed(2)}%</div>
                <small>年化波動率</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stock-info-card">
                <h6>RSI指標</h6>
                <div class="value ${stats.rsi < 30 ? 'change-positive' : stats.rsi > 70 ? 'change-negative' : ''}">
                    ${stats.rsi.toFixed(2)}
                </div>
                <small>${stats.rsi < 30 ? '超賣' : stats.rsi > 70 ? '超買' : '正常'}</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stock-info-card">
                <h6>布林帶位置</h6>
                <div class="value">${stats.bb_position}</div>
                <small>ATR: ${stats.atr.toFixed(2)}</small>
            </div>
        </div>
    `;
    
    document.getElementById('stockInfo').innerHTML = infoHtml;
}

function displaySignals(signals) {
    if (signals.length === 0) {
        document.getElementById('signals').innerHTML = `
            <div class="col-12">
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> 暫無明確交易信號
                </div>
            </div>
        `;
        return;
    }
    
    let signalsHtml = '';
    signals.forEach(signal => {
        const signalClass = signal.action.includes('買入') ? 'signal-buy' : 
                          signal.action.includes('賣出') ? 'signal-sell' : 'signal-neutral';
        
        signalsHtml += `
            <div class="col-md-4">
                <div class="signal-card ${signalClass}">
                    <h6><i class="fas fa-chart-line"></i> ${signal.indicator}</h6>
                    <p class="mb-1"><strong>信號:</strong> ${signal.signal}</p>
                    <p class="mb-0"><strong>建議:</strong> ${signal.action}</p>
                </div>
            </div>
        `;
    });
    
    document.getElementById('signals').innerHTML = signalsHtml;
}

function displayCharts(charts) {
    // 價格圖表
    const priceChartDiv = document.getElementById('priceChart');
    priceChartDiv.innerHTML = charts.price;
    
    // 布林帶圖表
    const bbChartDiv = document.getElementById('bbChart');
    bbChartDiv.innerHTML = charts.indicators;
    
    // ATR圖表
    const atrChartDiv = document.getElementById('atrChart');
    atrChartDiv.innerHTML = charts.atr;
    
    // 重新渲染Plotly圖表以確保正確顯示
    window.dispatchEvent(new Event('resize'));
}

function displayForecastTable(forecasts) {
    const tbody = document.getElementById('forecastBody');
    let html = '';
    
    for (let i = 0; i < forecasts.dates.length; i++) {
        const lr = forecasts.linear_regression[i] || 0;
        const arima = forecasts.arima[i] || 0;
        const lstm = forecasts.lstm[i] || 0;
        const avg = (lr + arima + lstm) / 3;
        
        html += `
            <tr>
                <td>${forecasts.dates[i]}</td>
                <td>$${lr.toFixed(2)}</td>
                <td>$${arima.toFixed(2)}</td>
                <td>$${lstm.toFixed(2)}</td>
                <td><strong>$${avg.toFixed(2)}</strong></td>
            </tr>
        `;
    }
    
    tbody.innerHTML = html;
}

// 初始化頁面加載完成後自動分析第一隻股票
document.addEventListener('DOMContentLoaded', function() {
    // 可以選擇在頁面加載時自動分析，或者等待用戶點擊
    // analyzeStock();
});

// 添加鍵盤快捷鍵支持
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeStock();
    }
});
