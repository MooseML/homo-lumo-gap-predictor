# Dockerfile for Hugging Face Space: Streamlit + RDKit + PyG

FROM python:3.10-slim

# System libraries (needed by RDKit / Pillow) 
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
    
# Create a non-root user to run the application
RUN useradd -m appuser

# Python packages 
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
    
# Set up working directory
WORKDIR /app

# Copy application files
COPY . .

# Fix permissions for temporary directories
RUN mkdir -p /tmp/streamlit && \
    chmod -R 777 /tmp && \
    chmod -R 777 /tmp/streamlit && \
    # Also ensure the SQLite database directory is writable
    mkdir -p /data && \
    chmod -R 777 /data && \
    # Make sure the app files are readable
    chmod -R 755 /app

# Ensure temp directories exist and are writable
RUN mkdir -p /tmp/csv_uploads && \
    chmod -R 777 /tmp/csv_uploads

# Set environment variables
ENV DB_DIR=/data \
    TMPDIR=/tmp \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_TELEMETRY_DISABLED=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Increase file upload size limit to accommodate larger CSVs
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50 
    
# Expose the port Streamlit will run on
EXPOSE 7860

# Set entrypoint script
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
echo "Starting Streamlit app with debug info"
echo "Current directory: $(pwd)"
echo "Files in current directory: $(ls -la)"
echo "Python version: $(python --version)"
echo "Temp directory: $TMPDIR"
echo "Temp directory exists: $([ -d $TMPDIR ] && echo 'Yes' || echo 'No')"
echo "Temp directory permissions: $(ls -ld $TMPDIR)"

# Run the app
streamlit run app.py
EOF

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Switch to the non-root user for better security
USER appuser

# Launch using the entrypoint script
CMD ["/app/entrypoint.sh"]