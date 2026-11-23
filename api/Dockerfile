FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

# No correr como root dentro del contenedor
RUN useradd --create-home api && chown -R api /app
USER api

# El contenedor siempre escucha en 8000; el puerto público (80XX) se define en docker-compose.yml
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=8)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
