// 在 analyzeStock 函數中修改 fetch
const response = await fetch('/api/analyze', {
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

// 更新股票列表函數
function updateStockList(market) {
    fetch(`/api/stocks?market=${market}`)
        .then(response => response.json())
        .then(stocks => {
            // ... 原有代碼
        });
}
