# Use the immutable base image
FROM hiroregistry/cogflowfltorch@sha256:cf55a95bca6487de597081dd4ea5ee62198092a56062f2b14debbca2d7c21471

# Upgrade CogFlow
RUN pip uninstall -y cogflow && pip install cogflow==1.11.25

# Optionally verify
RUN python3 -c 'import cogflow'

