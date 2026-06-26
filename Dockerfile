# Disaster Text Classifier — Production container
# Uses a slim Python image to keep the Cloud Run cold-start fast.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /service

# Create a non-root user for runtime security.
RUN useradd -m -u 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

# Ensure the non-root user can read the application files.
RUN chown -R appuser:appuser /service
USER appuser

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
