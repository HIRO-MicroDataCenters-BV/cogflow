# Use the official Python image as the base image
FROM python:3.10-slim

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG COGFLOW_VERSION=1.11.15

# Install specific version of cogflow
RUN pip3 install --ignore-installed cogflow==${COGFLOW_VERSION}

# Install additional Python packages (shap, xgboost).
# shap pinned to 0.44.* which publishes cp310 manylinux wheels; newer shap
# (0.47+) has a broken setup.py for the slim base and falls back to source
# builds that need g++.
RUN pip3 install "shap==0.44.*" xgboost

# Install compatible protobuf version first
RUN pip install --no-cache-dir protobuf==3.20.1

# Verify the installation of cogflow
RUN python3 -c 'import cogflow'
