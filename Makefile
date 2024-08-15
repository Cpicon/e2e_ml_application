# Default target if not provided
TARGET ?= dev

# Set DOCKER_COMPOSE_FILE based on the TARGET parameter
ifeq ($(TARGET),dev)
    DOCKER_COMPOSE_FILE=deployment/dev/compose.yaml
endif
ifeq ($(TARGET),stage)
    DOCKER_COMPOSE_FILE=deployment/stage/compose.yaml
endif
ifeq ($(TARGET),prod)
    DOCKER_COMPOSE_FILE=deployment/prod/compose.yaml
endif

# Build the docker image
build:
	@echo "Building docker images..."
	docker compose -f deployment/dev/compose.yaml build

# Run the docker container
run:
	@echo "Running docker containers..."
	docker compose -f $(DOCKER_COMPOSE_FILE) up --detach --wait

# Stop the docker container
stop:
	@echo "Stopping docker containers..."
	docker compose -f $(DOCKER_COMPOSE_FILE) stop

# Clean the docker container
clean:
	@echo "Cleaning up docker containers..."
	docker compose -f $(DOCKER_COMPOSE_FILE) down --remove-orphans --volumes


# Display help message
help:
	@echo "Makefile commands:"
	@echo "  make build - To build the docker images."
	@echo "  make run TARGET=<dev|stage|prod>     - To run the docker containers."
	@echo "  make stop TARGET=<dev|stage|prod>    - To stop the docker containers."
	@echo "  make clean TARGET=<dev|stage|prod>   - To clean up the docker containers."

.PHONY: build run stop clean help