# BookDB

BookDB is a book recommendation system that uses vector embeddings and collaborative filtering to suggest personalized reading recommendations to users. The application provides an intuitive interface for users to discover new books based on their reading history and preferences.

## Architecture

BookDB uses a microservices architecture with the following components:

- **Frontend**: React-based web application 
- **Backend**: Go API service
- **PostgreSQL**: SQL database for structured data
- **Qdrant**: Vector database for similarity search

All components are containerized using Docker for easy deployment and scalability.

## Dataset

Provide guidance on where to download the dataset and how to set it up for use with BookDB.

## Model

Add a description of the model that performs the best, based on your evaluation.

## Training and tuning

Provide instructions on training and tuning the model, along with the necessary compute and data requirements.

## Inference

- Add a description of how to use the model to make inferences.
- Provide an example of how a user would use BookDB to recommend the top three books for them to read.

## Design and Development

Ensure the engineering shines! Describe and justify the library design, and refer to the documentation.

## Deployment

BookDB is designed to be easily deployed using Docker containers.

### Deployment Options

- **Standard Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md) for direct Docker Compose deployment
- **Portainer Deployment**: See [DOCKER.md](DOCKER.md) for Portainer-specific instructions

### Infrastructure Requirements

- Docker and Docker Compose
- Minimum 2GB RAM for all services
- At least 10GB free disk space for container images and data volumes
- PostgreSQL and Qdrant databases (automatically deployed as containers)