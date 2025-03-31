# Start with a Node.js base image for the frontend build
FROM node:18 as frontend-builder

# Set working directory
WORKDIR /app

# Copy frontend source
COPY self-ui/ ./self-ui/

# Build frontend
WORKDIR /app/self-ui
RUN npm install
RUN npm run build

# Switch to a Python 3.10 base image for the backend
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY fast-api/ ./fast-api/

# Copy built frontend from the previous stage
COPY --from=frontend-builder /app/self-ui/build/ ./fast-api/frontend/

# Install Python dependencies
WORKDIR /app/fast-api
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Command to run the application
CMD gunicorn main:app -k uvicorn.workers.UvicornWorker --workers 1 --threads 16 --bind 0.0.0.0:${PORT:-8000}
