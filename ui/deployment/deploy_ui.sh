#!/bin/bash

# Now, we're going to authenticate Docker with GAR
echo "Configuring Docker for GAR authentication..."
export GOOGLE_APPLICATION_CREDENTIALS=$GCP_KEY
sudo gcloud auth activate-service-account --quiet --account pipeline@neural-needledrop.iam.gserviceaccount.com --key-file=$GOOGLE_APPLICATION_CREDENTIALS
sudo gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull the latest Docker image
echo "Pulling the latest Docker image..."
sudo docker pull us-central1-docker.pkg.dev/neural-needledrop/neural-needledrop-webapp/neural-needledrop-ui:latest

# Stop and remove the current running container
echo "Stopping and removing current container..."
sudo docker stop neural-needledrop-ui-container || true
sudo docker rm neural-needledrop-ui-container || true

# Run the UI container
echo "Running the UI container..."
sudo docker run -p 8080:8080 neural-needledrop-ui