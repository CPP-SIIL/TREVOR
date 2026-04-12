FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project so shared code works
COPY . .

EXPOSE 8000

CMD ["uvicorn", "dashboard.dashboard_api:app", "--host", "0.0.0.0", "--port", "8000"]