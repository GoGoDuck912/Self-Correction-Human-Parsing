# Human Parsing Docker Container
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    wget \
    libxcb1 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-randr0 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Install compatible numpy first
RUN pip install numpy==1.21.6

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install additional dependencies for this project
RUN pip install ninja tqdm pillow==8.4.0

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p inputs outputs models

# Set environment variables for compilation
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0"
ENV FORCE_CUDA="1"

# Expose port (if needed for any web interface)
EXPOSE 8000

# Default command
CMD ["python", "simple_extractor.py", "--help"]
