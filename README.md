# MNIST End to End ML model deployment
Efficiently deploy a machine learning model using containerized environments with data versioning, training, and live inference capabilities. Includes reproducible pipelines, monitoring, logging, and infrastructure automation to ensure scalable and robust performance for ML applications.

This repository contains the code and scripts to train, deploy, and serve a simple neural network model for the MNIST dataset. The solution is designed to be reproducible, scalable, and efficient, with infrastructure automation, monitoring, and logging.

## Environment Setup

The environment is set up using Docker to ensure reproducibility across different platforms.

### Steps:
1. Clone this repository:
    ```bash
    git clone git@github.com:Cpicon/e2e_ml_application.git
    cd e2e_ml_application
    ```
   
## Prerequisites

- Python 3.10
- Docker
- Poetry (Python dependency management tool)

## Setting Up the Environment

1. **Create a Virtual Environment:**
   - First, create a virtual environment using Python 3.10:
     ```bash
     python3.10 -m venv venv
     source venv/bin/activate
     ```

2. **Install Poetry:**
   - Install Poetry within the virtual environment:
     ```bash
     pip install poetry
     ```

3. **Install Project Dependencies:**
   - Use Poetry to install all the required packages:
     ```bash
     poetry install
     ```

4. **Set Environment Variables:**
   - Export the necessary environment variables for AWS and MLflow:
     ```bash
     export AWS_ACCESS_KEY_ID="awsaccesskey"
     export AWS_SECRET_ACCESS_KEY="awssecretkey"
     export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
     ```

## Building and Running Docker Containers

1. **Build Docker Images:**
   - Use `make` to build the Docker images:
     ```bash
     make build
     ```

2. **Run Docker Containers:**
   - After building the images, start the services using:
     ```bash
     make run
     ```
3. **Stop Docker Containers:**
   - To stop the services, use:
     ```bash
     make stop
     ```
4. **Remove Docker Containers:**
    - To remove the containers, use:
      ```bash
      make clean
      ```

## Using the Services

- **Dagster (Pipeline Orchestrator):**
  - Runs on [http://localhost:3000/](http://localhost:3000/).
  - Navigate to the "Overview/Jobs" section to view and manage your pipelines.

- **MLflow (Experiment Tracking and Model Registry):**
  - Available on [http://localhost:5005](http://localhost:5005).
  - Use MLflow to track your experiments, register models, and manage model versions.

- **Minio (Object and Model Storage):**
  - Runs on [http://localhost:9001](http://localhost:9001).
  - Minio serves as the object storage solution for the models and data.

- **MLServer (Model Deployment Service, HTTP Backend):**
  - Accessible at [http://localhost:9595](http://localhost:9595).
  - Use MLServer for deploying machine learning models with a RESTful API backend.

## Running and Training a Model

- To run and train a model, watch the following video tutorial: [![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](https://raw.githubusercontent.com/username/repository/branch/path/to/video.mp4)

## Deploying a Model

- To deploy a model, watch the following video tutorial: [![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](https://raw.githubusercontent.com/username/repository/branch/path/to/video.mp4)

## Testing the Deployed Model

- To test the deployed model, navigate to the root project folder and run:
  ```bash
  python model_query_example.py
    ```


## Architecture Overview

The architecture of this project is centered around several key components orchestrated using Docker containers. Below is a high-level overview of the services involved:

### 1. **Dagster**

- **Purpose**: Dagster is used as the pipeline orchestrator for managing and executing data pipelines.
- **Components**:
  - **Dagster PostgreSQL**: This service runs a PostgreSQL database for storing Dagster's run storage, schedule storage, and event logs.
  - **Dagster User Code**: This container runs the gRPC server that loads your user code, enabling Dagster to execute pipelines. It is configured to use the same image when launching runs in new containers.
  - **Dagster Webserver**: This service provides a web interface for interacting with Dagster, where you can view and manage pipelines.
  - **Dagster Daemon**: The daemon process is responsible for taking runs off of the queue and launching them, as well as handling schedules and sensors.

- **References**:
  - [What is Dagster?](https://dagster.io/blog/what-is-dagster)
  - [Dagster Code Locations](https://dagster.io/blog/dagster-code-locations)
  - [Dagster Logging](https://docs.dagster.io/concepts/logging#logging)

### 2. **MLflow**

- **Purpose**: MLflow is used for tracking experiments, registering models, and managing model versions.
- **Components**:
  - **MLflow PostgreSQL**: A PostgreSQL database for storing MLflow tracking metadata.
  - **MLflow Tracking Server**: This service hosts the MLflow server for tracking experiments and managing models.

### 3. **Minio**

- **Purpose**: Minio is used as an object storage solution for storing datasets, models, and other artifacts.
- **Components**:
  - **Minio Server**: A high-performance object storage server.
  - **Minio Client (mc)**: A command-line tool for interacting with Minio.

### 4. **MLServer**

- **Purpose**: MLServer is used for deploying machine learning models via a RESTful API.
- **Components**:
  - **MLServer**: The main service responsible for serving the machine learning models.

- **Reference**: [MLServer Documentation](https://mlserver.readthedocs.io/en/latest/index.html)

### Network and Volumes

- **Networks**: All services are connected through a `project_network` to facilitate communication between containers.
- **Volumes**: Persistent storage is managed using Docker volumes, ensuring that data persists across container restarts.


## Project Structure

- **deployment/**
  - Contains Docker-related configurations and deployment scripts.
  - **dev/**: Contains development-specific Dockerfiles, Docker Compose files, and configuration files for setting up the environment.
  - **prod/**: Contains production-specific Docker Compose file.
  - **stage/**: Contains staging-specific Docker Compose file.

- **e2eML/**
  - The main package directory for the project. It contains various submodules responsible for different stages of the machine learning lifecycle.
  - **clients/**: Contains code to interact with external services such as MLServer.
  - **evaluate/**: Handles the evaluation of machine learning models.
  - **inference/**: Contains scripts for making predictions using trained models.
  - **ingest/**: Responsible for data ingestion processes, such as loading datasets.
  - **models/**: Contains definitions for machine learning models.
  - **orchestrator/**: Contains code related to pipeline orchestration, likely tied to Dagster.
  - **pipeline_configs/**: Holds configuration files for various pipelines.
  - **train/**: Contains scripts and modules for training machine learning models.

