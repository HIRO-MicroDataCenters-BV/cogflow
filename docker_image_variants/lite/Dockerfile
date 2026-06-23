# Use the official Python image as the base image
FROM python:3.10-slim

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG COGFLOW_VERSION

# Install cogflow — pinned to COGFLOW_VERSION if supplied, otherwise latest from PyPI
RUN if [ -n "$COGFLOW_VERSION" ]; then \
      pip3 install --no-cache-dir --ignore-installed "cogflow==${COGFLOW_VERSION}"; \
    else \
      pip3 install --no-cache-dir --ignore-installed cogflow; \
    fi

# Verify the installation of cogflow
RUN python3 -c 'import cogflow; print("cogflow", cogflow.__version__)'
