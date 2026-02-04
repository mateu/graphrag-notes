FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG CUDA_COMPUTE_CAP=120
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH=/root/.cargo/bin:$PATH

# Build TEI from source with CUDA support
RUN git clone https://github.com/huggingface/text-embeddings-inference /opt/tei
WORKDIR /opt/tei
RUN cargo build --release --features "cuda cudarc/dynamic-linking" --bin text-embeddings-router

EXPOSE 80

ENTRYPOINT ["./target/release/text-embeddings-router"]
CMD ["--model-id", "jinaai/jina-embeddings-v3", "--port", "80"]
