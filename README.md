# MNIST End to End ML model deployment
Efficiently deploy a machine learning model using containerized environments with data versioning, training, and live inference capabilities. Includes reproducible pipelines, monitoring, logging, and infrastructure automation to ensure scalable and robust performance for ML applications.

This repository contains the code and scripts to train, deploy, and serve a simple neural network model for the MNIST dataset. The solution is designed to be reproducible, scalable, and efficient, with infrastructure automation, monitoring, and logging.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Handling](#data-handling)
3. [Model Training](#model-training)
4. [Model Deployment](#model-deployment)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Infrastructure Automation](#infrastructure-automation)
7. [Reproducing the Workflow](#reproducing-the-workflow)

## Environment Setup

The environment is set up using Docker to ensure reproducibility across different platforms.

### Steps:
1. Clone this repository:
    ```bash
    git clone git@github.com:Cpicon/e2e_ml_application.git
    cd e2e_ml_application
    ```

you can use the provided `Makefile` to build, run, stop, and clean up Docker containers for different environments (dev, stage, prod).

#### Steps:
1. Build the Docker image for the specified environment:
    ```bash
    make build TARGET=<dev|stage|prod>
    ```

2. Run the Docker container for the specified environment:
    ```bash
    make run TARGET=<dev|stage|prod>
    ```

3. Stop the Docker container for the specified environment:
    ```bash
    make stop TARGET=<dev|stage|prod>
    ```

4. Clean up the Docker containers and associated resources:
    ```bash
    make clean TARGET=<dev|stage|prod>
    ```

## Data Handling

The MNIST dataset is automatically downloaded and preprocessed as part of the pipeline. Data versioning is managed using a data versioning tool to ensure reproducibility.

### Steps:
1. The dataset is loaded and preprocessed in the `data_pipeline.py` script.
2. To ensure data versioning and reproducibility, the pipeline integrates with a data versioning tool configured in `dvc.yaml`.

## Model Training

The model is a simple neural network designed to classify MNIST digits. Training scripts are provided in the `train.py` file.

### Steps:
1. Train the model:
    ```bash
    python train.py
    ```
2. The trained model will be saved and versioned automatically.

## Model Deployment

The trained model is deployed using a lightweight and scalable serving platform, allowing it to handle live inference requests.

### Steps:
1. Deploy the model using the provided script:
    ```bash
    python deploy.py
    ```
2. The model will be served at the specified endpoint, ready to handle inference requests.

## Monitoring and Logging

Logging and monitoring are implemented to track the performance and health of the model during both training and inference.

### Features:
- Training and inference logs are captured using a logging framework.
- Basic monitoring includes metrics such as request rate, latency, and error rates.

## Infrastructure Automation

Infrastructure as Code (IaC) is used to automate the setup of cloud infrastructure, ensuring a seamless deployment experience.

### Steps:
1. Use the provided IaC scripts to set up the required cloud infrastructure:
    ```bash
    terraform apply
    ```
2. The infrastructure will be provisioned automatically, ready for model deployment.

## Reproducing the Workflow

To reproduce the entire workflow, follow these steps:

1. **Set up the environment:** Build and run the Docker container.
2. **Handle the data:** Run the data pipeline script to load and preprocess the dataset.
3. **Train the model:** Execute the training script to train and save the model.
4. **Deploy the model:** Use the deployment script to serve the model.
5. **Monitor and log:** Check logs and monitor metrics during training and inference.
6. **Automate the infrastructure:** Use the IaC tools to provision necessary cloud resources.

### Additional Information
- Ensure all dependencies are listed in the `requirements.txt` file.
- Use the `Makefile` to streamline common tasks such as building, training, and deploying.
