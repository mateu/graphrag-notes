FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
 && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone https://github.com/huggingface/text-embeddings-inference /opt/tei
WORKDIR /opt/tei

# Compile for RTX 50-series (compute capability 12.0)
# Set CUDA_COMPUTE_CAP to avoid build-time nvidia-smi dependency inside container.
ENV CANDLE_CUDA_ARCH=120
ENV CUDA_COMPUTE_CAP=120

RUN cargo build --release --features "cuda cudarc/dynamic-linking" --bin text-embeddings-router

EXPOSE 80

ENTRYPOINT ["/opt/tei/target/release/text-embeddings-router"]
