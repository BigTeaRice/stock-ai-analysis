# ---- 构建阶段 ----
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ---- 运行阶段 ----
FROM python:3.11-slim
WORKDIR /app
# 复制已编译的依赖
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
# 复制源码
COPY . .
# 非 root 运行
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8080
# gunicorn 启动（单 worker，Serverless 场景）
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "60", "api.index:app"]
