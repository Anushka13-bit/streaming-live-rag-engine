FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY corpus ./corpus
COPY harness ./harness

ENV OLLAMA_HOST=http://ollama:11434
ENV PRISM_CHAT_MODEL=llama3.2
ENV PRISM_EMBED_MODEL=nomic-embed-text

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
