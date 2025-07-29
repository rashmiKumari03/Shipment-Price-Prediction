# Use a slim base image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install system dependencies (in one layer, and clean up immediately)
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first (leverage Docker layer caching)
COPY requirements.txt .

# Install Python dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application
COPY . .

# Expose the port (optional, but good practice if our app uses it)
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
