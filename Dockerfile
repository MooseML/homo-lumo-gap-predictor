#  Dockerfile for Hugging Face Space: Streamlit + RDKit + PyG

FROM python:3.10-slim

#  system libraries (needed by RDKit / Pillow) 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libxrender1 \
        libxext6 \
        libsm6 \
        libx11-6 \
        libglib2.0-0 \
        libfreetype6 \
        libpng-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*
    
#  Python packages 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        streamlit==1.45.0 \
        rdkit-pypi==2022.9.5 \
        pandas==2.2.3 \
        numpy==1.26.4 \
        torch==2.2.0 \
        torch-geometric==2.5.2 \
        ogb==1.3.6 \
        pillow==10.3.0
    
#  working directory & app code 
WORKDIR /app
COPY . .
    
#  Streamlit configuration for Spaces 
ENV \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_TELEMETRY_DISABLED=true
    
EXPOSE 7860
    
#  launch 
CMD ["streamlit", "run", "app.py"]
    