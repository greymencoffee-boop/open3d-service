FROM python:3.11-slim

WORKDIR /app

# System deps for Open3D
RUN apt-get update && apt-get install -y \
    libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# v3 cache bust - force full pip reinstall
RUN echo "cache-bust-v3"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000
# Use shell form so $PORT expands at runtime
CMD ["/bin/sh", "-c", "exec python main.py"]
