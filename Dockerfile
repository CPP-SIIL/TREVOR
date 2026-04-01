# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt


# Command to run your main program
CMD ["python", "main.py"]