# Drowsiness Detection API

This is a FastAPI-based drowsiness detection service that uses computer vision and machine learning to detect driver drowsiness from images or video streams.

## Features

- Real-time drowsiness detection via WebSocket
- Image-based drowsiness detection via REST API
- Facial landmark detection
- Eye aspect ratio (EAR) calculation
- Mouth aspect ratio (MAR) calculation for yawning detection
- Emotion detection using DeepFace
- CNN and Random Forest models for drowsiness classification

## Docker Setup

### Building the Docker Image

```bash
# Navigate to the project directory
cd path/to/drowsiness-dms

# Build the Docker image
docker build -t sshubham2004/okdriver:drowsiness-dms .
```

### Running the Docker Container

```bash
# Run the container
docker run -p 8000:8000 sshubham2004/okdriver:drowsiness-dms

# Alternatively, use docker-compose
docker-compose up
```

### Pushing to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push the image
docker push sshubham2004/okdriver:drowsiness-dms
```

## API Endpoints

- `GET /`: Root endpoint to check if the API is running
- `GET /health`: Health check endpoint
- `POST /api/detect/drowsiness`: Endpoint for image-based drowsiness detection
- `POST /api/alert`: Endpoint for sending alerts
- `WebSocket /ws/drowsiness`: WebSocket endpoint for real-time frame processing

## Usage Examples

### REST API

```python
import requests

# Send an image for drowsiness detection
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/detect/drowsiness', files=files)
    result = response.json()
    print(result)
```

### WebSocket

```javascript
// JavaScript example for WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws/drowsiness');

ws.onopen = () => {
    console.log('Connected to WebSocket');
    // Send base64 encoded image
    ws.send(base64Image);
};

ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log('Drowsiness detection result:', result);
};
```

## Dependencies

See `requirements_api.txt` for the full list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.