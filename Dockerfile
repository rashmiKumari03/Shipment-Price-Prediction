FROM python:3.9-slim

# Install only what's necessary and clean up afterward
RUN apt-get update && \
    apt-get install -y awscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only needed files
COPY . .

# Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Run the app
CMD ["python3", "app.py"]
