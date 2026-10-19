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

# TensorFlow variant (layers on the main image of the same version)
docker build --build-arg COGFLOW_VERSION=3.0.0b15 -t hiroregistry/cogflow:3.0.0b15-tensor docker_image_variants/tensor

# PyTorch variant (layers on the main image of the same version)
docker build --build-arg COGFLOW_VERSION=3.0.0b15 -t hiroregistry/cogflow:3.0.0b15-torch docker_image_variants/torch

🚀 Pushing to Registry

Once built, push your images to your internal registry (hiroregistry):

docker push hiroregistry/cogflow:1.11.14-lite
docker push hiroregistry/cogflow:1.11.14-fl
docker push hiroregistry/cogflow:1.11.14-tensor
docker push hiroregistry/cogflow:1.11.14-torch

---

## 🚀 Release CI/CD & architecture policy

The `Build and Release` workflow (`.github/workflows/release.yml`) builds and
publishes the `cogflow` (full) and `cogflow_lite` images in the `docker-publish`
job. The policy decisions encoded there are documented here so they stay
discoverable — the workflow comments only point back to this section.

### Triggers

The `docker-publish` job runs in two modes:

- **Tag push (`v*`)** — the `build` and `publish-pypi` jobs run first; the job
  gates on `publish-pypi` succeeding.
- **`workflow_dispatch`** — a manual rebuild of a version already on PyPI.
  `build`/`publish-pypi` are skipped (their `if` requires a push event), so the
  job uses `always()` plus an inner condition to still run and pick the mode.

### Channel & tags

The channel is derived from the version suffix:

| Version suffix | Channel | Moving tag       |
|----------------|---------|------------------|
| `*b*`          | beta    | `latest-beta`    |
| `*rc*`         | rc      | `latest-rc`      |
| `*a*`          | alpha   | `latest-alpha`   |
| (none)         | stable  | `latest`         |

Only **stable** moves the plain `latest` tag; pre-release channels move their
own `latest-<channel>` tag so a beta never overwrites `latest`.

### Architecture policy

| Channel     | Platforms                 | Why |
|-------------|---------------------------|-----|
| stable      | `linux/amd64,linux/arm64` | Consumer-facing (main-branch images, downstream charm releases, external users) — must work on arm64. |
| pre-release | `linux/amd64`             | Consumed only by our own dev-cluster CD, which runs on amd64 nodes. Skipping arm64 saves ~8–10 min per release (QEMU emulation is the slow step). |

`platforms` is passed **explicitly even for amd64-only builds**: otherwise
`docker/build-push-action@v6` wraps the output in a manifest index that
downstream `docker pull --platform linux/arm64` consumers reject outright. QEMU
is only set up when the platform list contains a non-native arch (arm64), and it
reads the same `platforms` output so the QEMU and build lists can't drift.

`cogflow_lite` reuses this same platform policy. It has no compiled heavy deps
(no shap/xgboost/torch), so its arm64 build is cheap, and matching the full
image's platforms keeps it runnable both on the dev cluster (amd64) and for
arm64 stable consumers.

### Build strategy: wheel artifact vs PyPI

- **Tag push:** the `build` job uploads the freshly built wheel as an artifact.
  `docker-publish` downloads it into `docker_image_variants/latest/dist/` and the
  `latest` Dockerfile installs from that local wheel — no PyPI round-trip, so the
  image can't race CDN propagation of a just-uploaded version.
- **`workflow_dispatch`:** no artifact exists (the `build` job was skipped), so
  `dist/` is left empty and the Dockerfile falls back to
  `pip install cogflow==$VERSION` from PyPI. A poll of the PyPI simple index runs
  first as a safety net for propagation lag.
- **`cogflow_lite`:** always installs `cogflow` from PyPI (its Dockerfile has no
  wheel bind-mount). On tag-push it therefore needs its own simple-index wait
  before building — the wheel was published seconds earlier by `publish-pypi`. On
  `workflow_dispatch` the earlier dispatch wait already covers it.

### Training-job variants (`torch` / `tensor`)

`docker_image_variants/torch` and `docker_image_variants/tensor` are the images
the platform's Kubeflow training jobs run in (`PyTorchJob` → `-torch`, `TFJob` →
`-tensor`; entrypoints `python -m cogflow.trainers.<framework>.<name>`, submitted
by CogAPI). They are built in `docker-publish` right after the main image's smoke
test:

| Variant | Base (runtime image) | Adds per release | Tags |
|---------|----------------------|------------------|------|
| torch   | `hiroregistry/cogflow-runtime:torch-<TORCH_VERSION>` (python 3.10 + gcc + CUDA-enabled `torch`) | cogflow wheel, shap, xgboost, protobuf pin | `<version>-torch`, `latest[-<channel>]-torch` |
| tensor  | `hiroregistry/cogflow-runtime:tensor-<TF_VERSION>` (python 3.10 + gcc + `tensorflow[and-cuda]`) | cogflow wheel, shap, xgboost, `protobuf==3.20.3` re-pin | `<version>-tensor`, `latest[-<channel>]-tensor` |

Design points:

- **Framework first, cogflow on top.** The multi-GB framework layers live in
  the *runtime base* images (`runtime/torch`, `runtime/tensor`), which carry no
  cogflow and are rebuilt only when their Dockerfile or the pinned version in
  `runtime/versions.env` changes (`.github/workflows/runtime-images.yml`; a
  published version tag is never rebuilt unless forced). The per-release
  variant adds only the cogflow wheel plus light extras, so a cogflow release
  re-pushes a few MB and a node that has the base pulls only that delta.
  Previously the variants were `FROM hiroregistry/cogflow:<version>`, which put
  the 5–7 GB framework layers *above* a layer that changed on every release.
- **Bumping a framework**: edit `runtime/versions.env` (and the Dockerfile if
  needed), merge — `runtime-images.yml` publishes the new base — then the next
  cogflow release builds its variants from it. The variants' `RUNTIME_IMAGE`
  build-arg is derived from the same file by `release.yml`.
- The variants are built from the `docker_image_variants/` context with an
  explicit `file:` so their wheel bind-mount reads `latest/dist/` (the same
  wheel the main image installs; PyPI fallback on a dispatch rebuild).
- **amd64-only on every channel** (unlike the main image, which is multi-arch on
  stable): the CUDA torch wheels and `tensorflow[and-cuda]` have no arm64 build.
  One image serves `gpus=0` and `gpus>0` jobs; the price is size (several GB,
  pulled once per node).
- **TensorFlow stays on the 2.17 line** because it is the last line that
  accepts the `protobuf 3.20.x` the vendored kfp needs; 2.18+ requires
  protobuf ≥ 5. Multi-worker TensorFlow training is not supported (Keras 3
  cannot `fit` under `MultiWorkerMirroredStrategy`); the trainer refuses it.
- Each variant has its own cache scope (`torch`, `tensor`).
- A variant build failure leaves the main image published but that variant
  un-pushed — re-run `docker-publish` to reconcile (same caveat as `cogflow_lite`).
- Versions that predate `cogflow.trainers` (< 3.0.0b15) get no variants: the job
  probes the pushed main image for the package and skips the variant steps on a
  `workflow_dispatch` rebuild of such a version.

### Build cache

Each image uses its own GitHub Actions cache scope (`scope=latest` for the full
image, `scope=lite` for lite). They have different Dockerfiles, so sharing a
scope would make them evict each other's layers.

### Release ordering caveat

`cogflow_lite` is built after the full image is already pushed and smoke-tested.
A failure in the lite build (PyPI timeout, arm64 build error) therefore leaves
the full image published but lite un-pushed. Re-run the `docker-publish` job once
the cause is resolved to reconcile.