FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib TORCH_HOME=/opt/torch
COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e . && python -c "from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights; wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)"
COPY configs ./configs
RUN useradd --create-home appuser && mkdir -p artifacts && chown appuser artifacts
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"
CMD ["uvicorn", "anomaly.api:create_app", "--factory", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
