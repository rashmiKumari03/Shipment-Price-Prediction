# Using a lightweight Python image , Smaller images download and build faster.
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching , So that Docker will reuse the already-installed dependencies if requirements.txt hasn't changed. 
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \                          
    rm -rf /var/lib/apt/lists/*         

# Copy the rest of the application code
COPY . .

# Expose the port your app runs on
EXPOSE 8080

# Start the application
CMD ["python3", "app.py"]
