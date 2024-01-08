# Use a multi-stage build to keep image size down
# First stage: build
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git

# Set work directory
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy pyproject.toml and poetry.lock (if available)
COPY pyproject.toml poetry.lock* /app/

# Export the dependencies to a requirements.txt file and install them globally
RUN poetry export --without-hashes --format=requirements.txt > poetry-requirements.txt
RUN pip install --no-cache-dir -r poetry-requirements.txt

# Copy requirements.txt and install additional dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the run_pipeline.py script
COPY run_pipeline.py /app/

# Second stage: runtime
FROM python:3.11-slim as runtime

# Install ffmpeg, and then clean up the apt cache
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Set environment variables
ENV GBQ_PROJECT_ID=neural-needledrop \
    GBQ_DATASET_ID=backend_data \
    PATH=/root/.local/bin:$PATH

# Set up Google Cloud authentication
COPY gcloud-service-key.json* /app/
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcloud-service-key.json

# Set the OpenAI API key securely
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Set work directory
WORKDIR /app

# Copy the entire pipeline directory
COPY . /app/

# Run the pipeline script
CMD ["python", "run_pipeline.py"]
