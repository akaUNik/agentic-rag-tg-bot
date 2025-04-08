#!/bin/bash

# Stop the running container if it exists
echo "Stopping the agentic_rag_bot container..."
docker stop agentic_rag_bot || echo "No running container to stop."

# Remove the container if it exists
echo "Removing the agentic_rag_bot container..."
docker rm agentic_rag_bot || echo "No container to remove."

# Pull the latest changes from the repository
echo "Pulling the latest changes from Git..."
git pull || { echo "Git pull failed. Aborting."; exit 1; }

# Build the Docker image
echo "Building the Docker image for agentic_rag_bot..."
docker build -t agentic_rag_bot . || { echo "Docker build failed. Aborting."; exit 1; }

# Run the Docker container
echo "Starting the agentic_rag_bot container..."
docker run -d \
--name agentic_rag_bot \
--restart unless-stopped \
--env-file .env \
agentic_rag_bot || { echo "Failed to start the container. Aborting."; exit 1; }

# Tail the container logs
echo "Tailing the logs for agentic_rag_bot..."
docker logs -f agentic_rag_bot