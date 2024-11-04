# Use hiroregistry/cogflow:dev as the base image
FROM hiroregistry/cogflow:dev

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install specific version of cogflow
RUN pip3 install --ignore-installed cogflow==1.9.44

# Install additional Python packages (shap, xgboost)
RUN pip3 install --ignore-requires-python shap xgboost

# Verify the installation of cogflow
RUN python3 -c 'import cogflow'
