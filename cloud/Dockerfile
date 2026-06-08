FROM python:3.10-slim

WORKDIR /app

COPY cloud/requirements.txt /tmp/cloud-requirements.txt
RUN pip install --no-cache-dir -r /tmp/cloud-requirements.txt

COPY cloud /app/cloud

EXPOSE 8000

CMD ["uvicorn", "cloud.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
