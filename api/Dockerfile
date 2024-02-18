# STAGE 1: BUILD STAGE
FROM python:3.11-slim as builder

# Set the working directory in the builder stage
WORKDIR /usr/src/app

# Install Poetry
RUN pip install poetry

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY pyproject.toml poetry.lock ./

# Create a requirements.txt file from the pyproject.toml file
RUN poetry export --only main --with-credentials --without-hashes -o requirements.txt

# Install the application's dependencies
RUN pip install -r requirements.txt

# Copy the rest of the files to the working directory
COPY . .

# STAGE 2: RUNTIME STAGE
# Use a slim version of Python for this stage
FROM python:3.11-slim

# Set the working directory in the runtime stage
WORKDIR /usr/src/app

# Copy the Python environment from the builder stage
COPY --from=builder /usr/local /usr/local

# Copy the rest of the files to the working directory
COPY --from=builder /usr/src/app /usr/src/app

# Set the OpenAI API key securely
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Expose the port that the application runs on
EXPOSE 8000

# Keep the container running
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]




