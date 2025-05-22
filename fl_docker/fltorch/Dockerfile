# Use the official Python image as the base image
FROM python:3.10-slim

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install specific version of cogflow

RUN pip3 install --ignore-installed cogflow==1.10.7

# Install additional Python packages
RUN pip3 install \
    --no-cache-dir \
    shap \
    xgboost \
    flwr==1.8.0 \
    flwr_datasets \
    torch \
    torchvision


# Install compatible protobuf version first
RUN pip install --no-cache-dir protobuf==3.20.3

# Verify the installation of cogflow
RUN python3 -c 'import cogflow'
