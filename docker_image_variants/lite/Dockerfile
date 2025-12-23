# Use the official Python image as the base image
FROM python:3.10-slim

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install specific version of cogflow
RUN pip3 install --ignore-installed cogflow==1.11.2

# Verify the installation of cogflow
RUN python3 -c 'import cogflow'
