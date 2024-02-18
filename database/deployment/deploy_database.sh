#!/bin/bash

# Now, we're going to authenticate Docker with GAR
echo "Configuring Docker for GAR authentication..."
export GOOGLE_APPLICATION_CREDENTIALS=$GCP_KEY
sudo gcloud auth activate-service-account --quiet --account pipeline@neural-needledrop.iam.gserviceaccount.com --key-file=$GOOGLE_APPLICATION_CREDENTIALS
sudo gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull the ankane/pgvector Docker image
echo "Pulling the latest Docker image..."
sudo docker pull ankane/pgvector:v0.5.1

# Stop and remove the current running container
echo "Stopping and removing current container..."
sudo docker stop neural-needledrop-database-container || true
sudo docker rm neural-needledrop-database-container || true

# Create the database data directory if it doesn't exist
echo "Creating the database data directory..."
sudo mkdir -p /database/data

# Run the database container
echo "Running the database container..."
sudo docker run --name neural-needledrop-database-container \
    -e POSTGRES_PASSWORD=my_password \
    -e POSTGRES_DB=neural_needledrop_data \
    -e POSTGRES_USER=my_user \
    -p 5432:5432 \
    -v /database/data:/var/lib/postgresql/data \
    -d ankane/pgvector:v0.5.1
