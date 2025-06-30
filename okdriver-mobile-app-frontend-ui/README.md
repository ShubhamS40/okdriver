# OK Driver - Driver Drowsiness Detection System

A real-time driver drowsiness detection system that uses computer vision to monitor driver alertness and prevent accidents caused by drowsy driving.

## Features

- Real-time drowsiness detection using facial landmarks
- Eye aspect ratio (EAR) calculation to detect drowsiness
- WebSocket API for real-time communication
- REST API for standard HTTP requests
- Mobile-friendly design
- Alerts when drowsiness is detected
- Configurable detection parameters

## System Architecture

The system consists of two main components:

1. **Backend**: Python-based FastAPI server that handles drowsiness detection
2. **Frontend**: Mobile app that captures video and displays alerts (created separately)

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- OpenCV
- dlib with Python bindings
- FastAPI
- WebSockets support

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/okdriver.git
cd okdriver
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the shape predictor model

The system requires the dlib facial landmark predictor model. Create a `backend/models` directory and download the model:

```bash
mkdir -p backend/models
# Option 1: Download directly
curl -L "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" | bzip2 -d > backend/models/shape_predictor_68_face_landmarks.dat

# Option 2: Manual download
# Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract it and place in the backend/models directory
```

## Running the Application

### Start the backend server

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`.

### API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Configuration settings can be adjusted in the `backend/config.py` file or by setting environment variables:

- `DROWSY_THRESHOLD`: Eye aspect ratio threshold (default: 0.25)
- `DROWSY_CONSECUTIVE_FRAMES`: Number of consecutive frames to trigger alert (default: 15)
- `DETECTION_INTERVAL`: Minimum time between alerts in seconds (default: 3.0)
- `DEBUG_MODE`: Enable/disable debug mode (default: true)
- `SAVE_DEBUG_FRAMES`: Save annotated frames for debugging (default: false)

## Testing

Run the test suite to verify functionality:

```bash
pytest
```

## API Endpoints

### REST API

- `GET /`: Root endpoint
- `GET /status`: Server status
- `POST /detect`: Detect drowsiness from a single frame

### WebSocket API

- `WebSocket /ws/{client_id}`: Real-time detection connection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- dlib for the facial landmark detection
- OpenCV for image processing
- FastAPI for the API framework
