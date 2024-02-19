#!/bin/bash

# Now, we're going to authenticate Docker with GAR
echo "Configuring Docker for GAR authentication..."
export GOOGLE_APPLICATION_CREDENTIALS=$GCP_KEY
sudo gcloud auth activate-service-account --quiet --account pipeline@neural-needledrop.iam.gserviceaccount.com --key-file=$GOOGLE_APPLICATION_CREDENTIALS
sudo gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull the latest Docker image
echo "Pulling the latest Docker image..."
sudo docker pull us-central1-docker.pkg.dev/neural-needledrop/neural-needledrop-webapp/neural-needledrop-api:latest

# Stop and remove the current running container
echo "Stopping and removing current container..."
sudo docker stop neural-needledrop-api-container || true
sudo docker rm neural-needledrop-api-container || true

# Run the API container
echo "Running the API container..."
sudo docker run -e POSTGRES_HOST=neural-needledrop-database -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:8000 --name neural-needledrop-api-container -d us-central1-docker.pkg.dev/neural-needledrop/neural-needledrop-webapp/neural-needledrop-api:latest
