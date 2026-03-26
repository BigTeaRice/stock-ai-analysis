document.addEventListener('DOMContentLoaded', function () {
    // ==========================================================
    // 1. 初始化全局变量和 DOM 元素
    // ==========================================================
    const stockCodeInput = document.getElementById('stockCode');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const marketSelect = document.getElementById('marketSelect');
    const predictDaysInput = document.getElementById('predictDays');
    const dataPeriodSelect = document.getElementById('dataPeriod');

    // 用于存储 ECharts 实例，防止重复初始化
    let kLineChart = null;
    let forecastChart = null;
    let factorChart = null;

    // ==========================================================
    // 2. 初始化 ECharts 图表容器
    // ==========================================================
    function initCharts() {
        // K线图容器
        const kLineDom = document.getElementById('kLineChart');
        kLineChart = echarts.init(kLineDom);

        // 预测图容器
        const forecastDom = document.getElementById('forecastChart');
        forecastChart = echarts.init(forecastDom);

        // Alpha因子雷达图容器
        const factorDom = document.getElementById('factorChart');
        factorChart = echarts.init(factorDom);
    }

    // ==========================================================
    // 3. 模拟数据或渲染空状态 (可选)
    // ==========================================================
    function renderEmptyState() {
        kLineChart.setOption({
            title: {
                text: '请选择股票并点击分析',
                left: 'center',
                top: 'center',
                textStyle: { color: '#999', fontSize: 16 }
            },
            backgroundColor: '#f8f9fa'
        });
        forecastChart.setOption({
            title: { text: '等待预测数据...', left: 'center', top: 'center', textStyle: { color: '#999' } },
            backgroundColor: '#f8f9fa'
        });
        factorChart.setOption({
            title: { text: '等待因子分析...', left: 'center', top: 'center', textStyle: { color: '#999' } },
            backgroundColor: '#f8f9fa'
        });
    }

    // ==========================================================
    // 4. 核心逻辑：发起分析请求
    // ==========================================================
    analyzeBtn.addEventListener('click', async function () {
        const market = marketSelect.value;
        const code = stockCodeInput.value.trim();
        const days = predictDaysInput.value;
        const period = dataPeriodSelect.value;

        if (!code) {
            alert('请输入股票代码！');
            return;
        }

        // 显示加载状态
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 分析中...';

        try {
            // 请求后端 API
            // 注意：如果你的前端和后端在不同端口，可能需要配置代理或使用绝对路径
            const response = await fetch(`/api/analysis/${code}`);
            if (!response.ok) {
                throw new Error('网络响应错误');
            }
            const data = await response.json();

            // 渲染数据
            updateUIWithAnalysis(data);

        } catch (error) {
            console.error('分析失败:', error);
            alert('分析失败，请检查控制台日志或网络连接。');
        } finally {
            // 恢复按钮状态
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-robot"></i> 開始AI分析';
        }
    });

    // ==========================================================
    // 5. 渲染数据到页面
    // ==========================================================
    function updateUIWithAnalysis(data) {
        // 5.1 更新基本数据卡片
        document.getElementById('currentPrice').innerText = `$${data.current_price.toFixed(2)}`;
        document.getElementById('volatility').innerText = `${data.volatility.toFixed(2)}%`;
        document.getElementById('rsi').innerText = data.rsi;
        document.getElementById('volume').innerText = `${(data.volume / 1000000).toFixed(2)}M`;

        // 5.2 渲染 K线图 (使用模拟数据格式)
        renderKLineChart(data.kline_data);

        // 5.3 渲染 AI 预测图
        renderForecastChart(data.forecast_data);

        // 5.4 渲染 Alpha 因子雷达图
        renderFactorRadar(data.factor_data);
    }

    // 5.5 渲染 K线图
    function renderKLineChart(klineData) {
        // 假设 klineData 是一个数组，包含 [时间, 开盘, 收盘, 最低, 最高] 的数据
        kLineChart.setOption({
            title: { text: '历史行情 (K线图)', left: 'left' },
            tooltip: { trigger: 'axis' },
            xAxis: { type: 'category', data: klineData.dates },
            yAxis: { type: 'value', name: '价格' },
            series: [{
                name: '收盘价',
                type: 'candlestick',
                data: klineData.values,
                itemStyle: {
                    color: '#ec0000',
                    color0: '#00da00',
                    borderColor: '#ec0000',
                    borderColor0: '#00da00'
                }
            }],
            backgroundColor: '#ffffff'
        });
    }

    // 5.6 渲染 AI 预测图
    function renderForecastChart(forecastData) {
        forecastChart.setOption({
            title: { text: 'AI 预测走势', left: 'left' },
            tooltip: { trigger: 'axis' },
            xAxis: { type: 'category', data: forecastData.dates },
            yAxis: { type: 'value', name: '预测价格' },
            series: [{
                name: '预测价格',
                type: 'line',
                data: forecastData.values,
                smooth: true,
                lineStyle: { width: 3, color: '#5470C6' }
            }],
            backgroundColor: '#ffffff'
        });
    }

    // 5.7 渲染 Alpha 因子雷达图
    function renderFactorRadar(factorData) {
        factorChart.setOption({
            title: { text: 'Alpha 因子分析', left: 'left' },
            radar: {
                indicator: factorData.indicators
            },
            series: [{
                name: '因子评分',
                type: 'radar',
                data: factorData.scores
            }],
            backgroundColor: '#ffffff'
        });
    }

    // ==========================================================
    // 6. 初始化
    // ==========================================================
    initCharts();
    renderEmptyState();
});