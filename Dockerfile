# ============================================================
# Stage 1: Build CUDA extensions
# ============================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0"
ENV FORCE_CUDA=1

# Install Python 3.10 + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git ninja-build g++ \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /build

# Install PyTorch with CUDA 12.8 support (must come from pytorch index, not PyPI)
# Separate RUN per package to reduce peak memory during pip wheel decompression
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128 && \
    rm -rf /root/.cache/pip
RUN pip install torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    rm -rf /root/.cache/pip

# Install core dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Install CUDA extensions that need compilation
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/ashawkey/diff-gaussian-rasterization/

RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/camenduru/simple-knn/

RUN MAX_JOBS=2 PYTORCH3D_NO_NINJA=1 pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/facebookresearch/pytorch3d.git

RUN pip install --no-cache-dir --no-build-isolation \
    nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling

# Install remaining deps needed at runtime (trimmed: no tensorflow/gradio/pandas/tensorboard)
RUN pip install --no-cache-dir \
    transformers accelerate moviepy scikit-image \
    huggingface_hub pymcubes==0.1.6 jaxtyping typeguard \
    matplotlib

# chumpy stub: FLAME pickle files contain chumpy.ch.Ch objects (numpy array subclass).
# Real chumpy has a broken setup.py; we only need the class for pickle deserialization.
RUN python -c "\
import os; \
sp = '/usr/local/lib/python3.10/dist-packages/chumpy'; \
os.makedirs(sp, exist_ok=True); \
open(os.path.join(sp, '__init__.py'), 'w').write( \
    'from .ch import Ch\n' \
); \
open(os.path.join(sp, 'ch.py'), 'w').write( \
    'import numpy as np\n' \
    'class Ch(np.ndarray):\n' \
    '    def __new__(cls, *args, **kwargs):\n' \
    '        if args: return np.asarray(args[0]).view(cls)\n' \
    '        return np.array([]).view(cls)\n' \
    '    def __array_finalize__(self, obj): pass\n' \
    '    @property\n' \
    '    def r(self): return np.asarray(self)\n' \
)"

# Apply compiled_autograd.h patch for Windows guard bug
# (also needed on Linux to prevent potential issues with USE_CUDA guard)
RUN HEADER=$(python -c "import torch; print(torch.__file__.replace('__init__.py',''))"); \
    PATCH_FILE="${HEADER}include/torch/csrc/dynamo/compiled_autograd.h"; \
    if [ -f "$PATCH_FILE" ]; then \
      sed -i 's/#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))/#if defined(_WIN32)/' "$PATCH_FILE"; \
    fi

# Verify CUDA extensions compile
RUN python -c "import diff_gaussian_rasterization; print('diff-gs OK')"
RUN python -c "import simple_knn; print('simple-knn OK')"
RUN python -c "import pytorch3d; print('pytorch3d OK')"

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV XFORMERS_DISABLED=1
ENV PYTHONUNBUFFERED=1
ENV NUMBA_THREADING_LAYER=omp

# g++, python3.10-dev, and cuda-nvcc for nvdiffrast JIT compilation at first startup
# CUDA library dev packages cause version conflicts with runtime image, so we copy
# include headers from the builder stage instead
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    g++ python3.10-dev \
    cuda-nvcc-12-8 cuda-cudart-dev-12-8 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Copy CUDA include headers from builder for nvdiffrast JIT (cusparse.h, etc.)
COPY --from=builder /usr/local/cuda/include/ /usr/local/cuda/include/

# Copy all installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source only (model weights mounted at runtime)
COPY serve.py .
COPY app_lam.py .
COPY lam/ ./lam/
COPY vhap/ ./vhap/
COPY tools/ ./tools/
COPY external/ ./external/
COPY configs/inference/ ./configs/inference/
COPY configs/stylematte_config.json ./configs/
COPY assets/images/ ./assets/images/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Model weights must be mounted: -v /path/to/model_zoo:/app/model_zoo
ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
