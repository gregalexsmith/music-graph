FROM --platform=linux/arm64 python:3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    python3-dev \
    python3-numpy \
    python3-yaml \
    python3-six \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libtag1-dev \
    libsndfile1 \
    libchromaprint-dev \
    libeigen3-dev \
    libyaml-dev \
    # Must be libsamplerate-dev (not just libsamplerate0-dev):
    libsamplerate-dev \
    # If single-precision is in a separate package, you might need libfftw3f-dev:
    # On many distros, libfftw3-dev includes both double & single precision
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Optional: ensure your pip is up to date.
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY classifier.py .
VOLUME ["/app/audio", "/app/data"]

CMD ["python", "classifier.py"]