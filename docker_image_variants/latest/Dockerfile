# Use the official Python image as the base image
FROM python:3.10-slim

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install specific version of cogflow

RUN pip3 install --ignore-installed cogflow==1.11.16

# Install additional Python packages (shap, xgboost)
RUN pip3 install --ignore-requires-python shap xgboost

# Install compatible protobuf version first
RUN pip install --no-cache-dir protobuf==3.20.1

# Verify the installation of cogflow
RUN python3 -c 'import cogflow'
