#!/bin/bash

# Check if Docker is installed and install it if necessary
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."

    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    # curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    # # Add the repository to Apt sources:
    # echo \
    #     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
    # $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |
    #     sudo tee /etc/apt/sources.list.d/docker.list >/dev/null


    # FROM CHATGPT:
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    # Set up the stable repository
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null


    sudo apt-get update

    # Now, we'll install Docker
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

fi

# Now, we're going to authenticate Docker with GAR
echo "Configuring Docker for GAR authentication..."
export GOOGLE_APPLICATION_CREDENTIALS=$GCP_KEY
sudo gcloud auth activate-service-account --quiet --account pipeline@neural-needledrop.iam.gserviceaccount.com --key-file=$GOOGLE_APPLICATION_CREDENTIALS
sudo gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull the latest Docker image
echo "Pulling the latest Docker image..."
sudo docker pull us-central1-docker.pkg.dev/neural-needledrop/neural-needledrop-webapp/neural-needledrop-pipeline:latest

# Stop and remove the current running container
echo "Stopping and removing current container..."
sudo docker stop neural-needledrop-pipeline-container || true
sudo docker rm neural-needledrop-pipeline-container || true

# Copy the pipeline_cron file to /etc/cron.d/
echo "Copying pipeline_cron to /etc/cron.d/"
sudo cp /tmp/pipeline_cron /etc/cron.d/pipeline_cron

# Set appropriate permissions for the cron file
echo "Setting permissions for the cron file"
sudo chmod 644 /etc/cron.d/pipeline_cron
sudo chown root:root /etc/cron.d/pipeline_cron

# Restart cron to apply changes
echo "Restarting cron service to apply changes"
sudo systemctl restart cron
