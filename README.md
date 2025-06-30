# okDriver - Driver Safety App

## Overview

okDriver is a comprehensive mobile application designed to enhance driver safety through real-time monitoring and alerts. The app uses computer vision and machine learning to detect potential hazards and provide timely warnings to drivers.

## Features

### Drowsiness Detection
- Real-time monitoring of driver's face
- Alerts when signs of drowsiness are detected
- Detailed drowsiness analysis

### Pothole Detection
- Camera-based pothole detection on roads
- Real-time alerts when potholes are detected
- Information about pothole size and location
- Helps drivers avoid vehicle damage

### AI Voice Assistant
- Hands-free operation while driving
- Voice commands for app control
- Safety information and alerts

## Project Structure

```
├── okdriver-mobile-app-frontend-ui/  # Flutter mobile app
│   ├── lib/                         # Flutter source code
│   │   ├── models/                  # Data models
│   │   ├── screens/                 # UI screens
│   │   ├── services/                # Backend services
│   │   ├── theme/                   # App theme
│   │   ├── utils/                   # Utilities
│   │   └── widgets/                 # Reusable widgets
│   ├── assets/                      # App assets
│   └── pubspec.yaml                 # Flutter dependencies
└── pithole/                         # Python backend for pothole detection
    ├── large_pothole_webcam.py      # Standalone pothole detection script
    └── pothole_api.py               # FastAPI backend for pothole detection
```

## Setup Instructions

### Prerequisites
- Flutter SDK (3.0.0 or higher)
- Python 3.8 or higher
- Android Studio / Xcode

### Mobile App Setup
1. Navigate to the Flutter app directory:
   ```
   cd okdriver-mobile-app-frontend-ui
   ```

2. Install dependencies:
   ```
   flutter pub get
   ```

3. Run the app:
   ```
   flutter run
   ```

### Pothole Detection Backend Setup
1. Navigate to the pothole directory:
   ```
   cd pithole
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model:
   - Download the `kaggle&RDD.pt` model file and place it in the `pithole` directory

4. Run the FastAPI server:
   ```
   python pothole_api.py
   ```

## API Endpoints

### Pothole Detection API
- **Endpoint**: `/detect_pothole`
- **Method**: POST
- **Request Body**: JSON with base64 encoded image
- **Response**: JSON with pothole detection results

## Technologies Used

- **Frontend**: Flutter, Dart
- **Backend**: Python, FastAPI
- **Computer Vision**: OpenCV, YOLOv8
- **Machine Learning**: Ultralytics YOLOv8

## License

This project is licensed under the MIT License - see the LICENSE file for details.