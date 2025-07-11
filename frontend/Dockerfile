FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y chromium \
    && rm -rf /var/lib/apt/lists/*
COPY ./fonts/NotoSansJP-VariableFont_wght.ttf /app/fonts/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["streamlit", "run", "app.py"]
