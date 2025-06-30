FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    pkg-config \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies in stages for better build stability
# Core numerical libraries first
RUN pip install --no-cache-dir numpy==1.24.3 scipy==1.11.3

# Install dlib with specific version
RUN pip install --no-cache-dir dlib==19.24.0

# Install TensorFlow
RUN pip install --no-cache-dir tensorflow==2.13.0

# Install OpenCV
RUN pip install --no-cache-dir opencv-python==4.8.0.76

# Install FastAPI and web dependencies
RUN pip install --no-cache-dir \
    "fastapi<0.99.0" \
    "uvicorn<0.24.0" \
    "pydantic<2.0.0" \
    python-multipart==0.0.6 \
    websockets==11.0.3

# Install remaining dependencies
RUN pip install --no-cache-dir \
    joblib==1.3.2 \
    scikit-learn==1.3.0 \
    deepface==0.0.79 \
    pillow==10.1.0 \
    python-jose==3.3.0 \
    passlib==1.7.4

# Copy model files and application code
COPY final_drowsiness_model_leakyrelu2.h5 .
COPY drowsiness_ml_model.pkl .
COPY shape_predictor_68_face_landmarks.dat .
COPY drowsiness_api.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "drowsiness_api:app", "--host", "0.0.0.0", "--port", "8000"]