# Use a development image that includes CUDA build tools
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy repository first (assuming you've cloned it)
COPY . .

# Install Python dependencies (excluding Flash Attention)
# Create a modified requirements file without flash-attn
RUN grep -v "flash-attn" requirements.txt > requirements_modified.txt && \
    pip install --no-cache-dir -r requirements_modified.txt

# Install PEFT and Transformers for LoRA support
RUN pip install peft transformers

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"