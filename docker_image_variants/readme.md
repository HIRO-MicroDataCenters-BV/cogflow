🧠 CogFlow Docker Images

This directory contains Docker build definitions for multiple CogFlow image variants, designed to support different runtime and training environments such as Federated Learning, TensorFlow, and PyTorch.

🧩 Image Naming Convention

All CogFlow images follow a unified tagging structure:

hiroregistry/cogflow:<version>[-<variant>]


📂 Directory Structure
fl_docker/
│
├── fl/         # Federated Learning image
│   └── Dockerfile

├── latest/     # Latest image
│   └── Dockerfile
│
├── lite/       # Minimal runtime image
│   └── Dockerfile
│
├── tensor/     # TensorFlow variant
│   └── Dockerfile
│
├── torch/      # PyTorch variant
│   └── Dockerfile
│
└── readme.md   # This documentation

🧱 Building Images

Each variant has its own Dockerfile under fl_docker/<variant>/Dockerfile.

Build Commands
# Lite variant (minimal)
docker build -t hiroregistry/cogflow:1.11.14-lite -f fl_docker/lite/Dockerfile .

# Federated Learning variant
docker build -t hiroregistry/cogflow:1.11.14-fl -f fl_docker/fl/Dockerfile .

# TensorFlow variant
docker build -t hiroregistry/cogflow:1.11.14-tensor -f fl_docker/tensor/Dockerfile .

# PyTorch variant
docker build -t hiroregistry/cogflow:1.11.14-torch -f fl_docker/torch/Dockerfile .

🚀 Pushing to Registry

Once built, push your images to your internal registry (hiroregistry):

docker push hiroregistry/cogflow:1.11.14-lite
docker push hiroregistry/cogflow:1.11.14-fl
docker push hiroregistry/cogflow:1.11.14-tensor
docker push hiroregistry/cogflow:1.11.14-torch