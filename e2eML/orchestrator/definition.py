from dagster import (
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
)
from ..ingest import mnist as ingest_mnist
from ..train import mnist as train_mnist


all_assets = load_assets_from_modules([ingest_mnist, train_mnist])

# Define a job that will materialize the assets
mnist_data_job = define_asset_job(
    "MNIST_data",
    selection=[
        ingest_mnist.train_MNIST_data,
        ingest_mnist.test_MNIST_data,
    ],
)

mnist_visualize_job = define_asset_job(
    "MNIST_visualize",
    selection=[
        ingest_mnist.get_dataloader,
        ingest_mnist.print_data,
    ],
)

mnist_train_job = define_asset_job(
    "MNIST_train",
    selection=[
        train_mnist.train_model,
    ],
    config={
        "resources": {
            "mlflow": {
                "config": {
                    "experiment_name": "MNIST Experiment",
                    "mlflow_tracking_uri": "http://localhost:5005",
                    # env variables to pass to mlflow
                    "env": {
                        "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
                        "AWS_ACCESS_KEY_ID": "awsaccesskey",
                        "AWS_SECRET_ACCESS_KEY": "awssecretkey",
                    },
                    # env variables you want to log as mlflow tags
                    "env_to_tag": ["DOCKER_IMAGE_TAG"],
                    # key-value tags to add to your experiment
                    "extra_tags": {"super": "experiment"},
                }
            }
        }
    }
)

# Addition: a ScheduleDefinition the job it should run and a cron schedule of how frequently to run it
mnist_schedule = ScheduleDefinition(
    job=mnist_data_job,
    cron_schedule="0 * * * *",  # every hour
)

defs = Definitions(
    assets=all_assets,
    jobs=[mnist_data_job, mnist_train_job, mnist_visualize_job],
    schedules=[mnist_schedule],
)
