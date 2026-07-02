FROM python:3.10-slim

LABEL org.opencontainers.image.title="Fair AD Predictor"
LABEL org.opencontainers.image.description="Fair Alzheimer's Disease prediction app with per-sex calibration, SHAP explanations, and self-updating model via user corrections"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.authors="GenderBias_AD Project"
LABEL org.opencontainers.image.url="https://github.com/imlinstress/GenderBias_AD"
LABEL org.opencontainers.image.source="https://github.com/imlinstress/GenderBias_AD"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
