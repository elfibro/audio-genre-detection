# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    cmake \
    build-essential \
    libyaml-cpp-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    curl

# -- 1) Installer Numpy < 2 pour compatibilité avec Essentia
RUN pip3 install --no-cache-dir "numpy<2.0"

# -- 2) Installer essentia-tensorflow et autres dépendances Python
RUN pip3 install --no-cache-dir essentia-tensorflow
RUN pip3 install --no-cache-dir fastapi uvicorn requests python-multipart

# Create a working directory
WORKDIR /app

# Cloner le dépôt GitHub
RUN git clone https://github.com/elfibro/audio-genre-detection.git .

# Rendre le script download.sh exécutable et l'exécuter
RUN chmod +x download.sh
RUN ./download.sh

# Exposer le port 13400
EXPOSE 13400

# Commande par défaut pour lancer ton application FastAPI
CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "13400"]
