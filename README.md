# CogFlow

**CogFlow** is a modular, SDK-first machine learning workflow management system
built on **Kubeflow Pipelines**, **MLflow**, **Kubernetes**, and **MinIO**.

It provides a clean Python API for:
- building production-grade ML pipelines
- managing datasets and components
- orchestrating federated learning workflows
- enforcing consistent error handling and validation

CogFlow is designed for **real infrastructure**, not notebooks only.

---

## Why CogFlow?

Modern ML platforms are powerful but fragmented:

- Kubeflow Pipelines → great orchestration, weak ergonomics
- MLflow → experiment tracking, limited workflow control
- Kubernetes → powerful, but verbose and error-prone
- Federated learning → no standard orchestration layer

**CogFlow bridges these gaps** by providing:

- a stable Python SDK
- safe lazy-loading of heavy dependencies
- unified error handling
- infrastructure-aware abstractions
- zero circular imports

---

## Core Features

### 🧩 Pipeline Orchestration
- Lazy-loaded Kubeflow Pipelines client
- Safe pipeline compilation and submission
- Runtime environment injection
- Kubernetes service lifecycle management

### 📦 Component Management
- YAML-based component registry
- MinIO-backed component storage
- Automatic component registration
- Runtime-safe component loading

### 📊 Dataset Management
- Dataset registration and metadata retrieval
- Secure dataset downloads
- Silent deletes with strict error semantics
- Pluggable storage backends

### 🤝 Federated Learning
- Auto-generated FL pipelines
- Dynamic pipeline signatures
- Connector-based and dataspace-based workflows
- Region-aware scheduling with node selectors

### 🧠 Unified Error Handling
- Strongly typed error hierarchy
- Context-aware exception wrapping
- API-ready error serialization
- Zero silent failures

---

## Architecture Overview

CogFlow follows a **layered SDK architecture**:

cogflow/
├── core/
│ ├── pipelines/ # Kubeflow orchestration & FL pipelines
│ ├── datasets/ # Dataset lifecycle management
│ ├── components/ # Component registry & YAML handling
│ └── models/ # (future extension)
│
├── utils/
│ ├── common.py # UUIDs, paths, Kubernetes helpers
│ ├── network.py # HTTP utilities with retry & streaming
│ ├── storage.py # MinIO client abstraction
│ └── exceptions.py # Unified error framework
│
├── config.py
└── api.py