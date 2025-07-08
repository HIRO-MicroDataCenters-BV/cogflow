# Use the official Python image as the base image
FROM python:3.11.10-slim

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install specific version of cogflow

RUN pip3 install --ignore-installed cogflow==1.10.9

# Install additional Python packages (shap, xgboost)
RUN pip3 install --ignore-requires-python tensorflow xgboost

# Verify the installation of cogflow
RUN python3 -c 'import cogflow'
