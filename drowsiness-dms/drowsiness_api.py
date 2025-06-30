import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.optimizers import Adam
import dlib
import joblib
from scipy.spatial import distance
from deepface import DeepFace
import os
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("drowsiness_api")

# === Register mixed precision Policy for deserialization ===
from tensorflow.keras.mixed_precision import Policy

@register_keras_serializable()
class MixedPolicy(Policy):
    pass

tf.keras.utils.get_custom_objects()['Policy'] = MixedPolicy
tf.keras.utils.get_custom_objects()['Adam'] = Adam

# Initialize FastAPI app
app = FastAPI(title="Drowsiness Detection API", description="API for detecting driver drowsiness")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Models ===
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "final_drowsiness_model_leakyrelu2.h5")
rf_model_path = os.path.join(current_dir, "drowsiness_ml_model.pkl")
landmark_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")

logger.debug(f"Current directory: {current_dir}")
logger.debug(f"CNN model absolute path: {model_path}")
logger.debug(f"RF model absolute path: {rf_model_path}")
logger.debug(f"Landmark predictor absolute path: {landmark_path}")

cnn_model = None
rf_model = None
detector = None
predictor = None

try:
    logger.info(f"Loading CNN model from {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"CNN model file not found at {model_path}")
    else:
        cnn_model = load_model(model_path)
        logger.info("CNN model loaded successfully")
    
    logger.info(f"Loading RF model from {rf_model_path}")
    if not os.path.exists(rf_model_path):
        logger.error(f"RF model file not found at {rf_model_path}")
    else:
        rf_model = joblib.load(rf_model_path)
        logger.info("RF model loaded successfully")
    
    logger.info(f"Loading face detector and landmark predictor from {landmark_path}")
    if not os.path.exists(landmark_path):
        logger.error(f"Landmark predictor file not found at {landmark_path}")
    else:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(landmark_path)
        logger.info("Face detector and landmark predictor loaded successfully")
    
    if cnn_model and rf_model and detector and predictor:
        logger.info("All models loaded successfully")
    else:
        logger.warning("Some models failed to load. The API will use mock data for missing models.")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    # We'll continue and handle missing models in the endpoints

# === Landmark Indexes ===
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(60, 68))

# === EAR/MAR thresholds ===
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    if len(mouth) < 7:
        return 0
    A = distance.euclidean(mouth[2], mouth[6])  # vertical
    B = distance.euclidean(mouth[3], mouth[5])  # vertical
    C = distance.euclidean(mouth[0], mouth[4])  # horizontal
    return (A + B) / (2.0 * C)

def sanitize(value):
    if isinstance(value, (np.bool_, np.bool8)):
        return bool(value)
    elif isinstance(value, (np.integer,)):
        return int(value)
    elif isinstance(value, (np.floating,)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value


# Function to process image for drowsiness detection
def process_image(image):
    if cnn_model is None or rf_model is None or detector is None or predictor is None:
        # Return mock data if models aren't loaded
        return {
            "drowsy": False,
            "ear": 0.0,
            "mar": 0.0,
            "cnn_confidence": 0.0,
            "emotion": "unknown",
            "is_yawning": False,
            "message": "Models not fully loaded",
            "face_rects": [],
            "landmarks": {
                "left_eye": [],
                "right_eye": [],
                "mouth": [],
                "all": []
            }
        }

    
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        if len(faces) == 0:
            return {
                "drowsy": False,
                "ear": 0.0,
                "mar": 0.0,
                "cnn_confidence": 0.0,
                "emotion": "unknown",
                "is_yawning": False,
                "message": "No face detected",
                "face_rects": [],
                "landmarks": {
                    "left_eye": [],
                    "right_eye": [],
                    "mouth": [],
                    "all": []
                }
            }
        
        # Process the first face found
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Ensure coordinates are within image bounds
        x, y = max(0, x), max(0, y)
        if (y + h) >= gray.shape[0] or (x + w) >= gray.shape[1]:
            return {
                "drowsy": False,
                "ear": 0.0,
                "mar": 0.0,
                "cnn_confidence": 0.0,
                "emotion": "unknown",
                "is_yawning": False,
                "message": "Face partially outside frame",
                "face_rects": [{
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                }],
                "landmarks": {
                    "left_eye": [],
                    "right_eye": [],
                    "mouth": [],
                    "all": []
                }
            }
        
        # Extract face region
        face_img = gray[y:y+h, x:x+w]
        face_rgb = image[y:y+h, x:x+w]
        
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Extract eye and mouth landmarks
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH]
        
        # Calculate EAR and MAR
        avg_EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        
        # CNN Prediction
        cnn_label, cnn_confidence = 0, 0.0
        try:
            # Resize to 224x224 and convert to RGB (3 channels) as expected by the model
            cnn_input = cv2.resize(face_img, (224, 224))
            # Convert grayscale to RGB by repeating the channel 3 times
            cnn_input = cv2.cvtColor(cnn_input, cv2.COLOR_GRAY2RGB) / 255.0
            cnn_input = cnn_input.reshape(1, 224, 224, 3)
            cnn_pred = cnn_model.predict(cnn_input, verbose=0)
            cnn_label = np.argmax(cnn_pred)
            cnn_confidence = float(np.max(cnn_pred))
        except Exception as e:
            logger.error(f"CNN prediction error: {str(e)}")
            cnn_label, cnn_confidence = 0, 0.0
        
        # RF Prediction
        rf_label = 0
        try:
            rf_input = np.array([[avg_EAR, 0.3, 0, 2.5, 2.4, 2.6]])
            rf_raw_label = rf_model.predict(rf_input)[0]
            label_map = {'drowsy': 1, 'alert': 0}
            rf_label = label_map.get(str(rf_raw_label).lower(), 0)
        except Exception as e:
            logger.error(f"RF prediction error: {str(e)}")
            rf_label = 0
        
        # Final decision
        is_drowsy = cnn_label == 1 or rf_label == 1 or avg_EAR < EAR_THRESHOLD
        
        # Yawning detection based on MAR threshold
        is_yawning = mar > MAR_THRESHOLD
        
        # Emotion detection
        emotion = "neutral"
        try:
            results = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
            emotion = results[0]['dominant_emotion']
        except Exception as e:
            logger.error(f"Emotion detection error: {str(e)}")
        
        # Extract facial landmarks for visualization
        all_landmarks = []
        for i in range(68):
            point = landmarks.part(i)
            all_landmarks.append({"x": int(point.x), "y": int(point.y)})
        
        # Return enhanced results
        return {
            "drowsy": is_drowsy,
            "ear": float(avg_EAR),
            "mar": float(mar),
            "cnn_confidence": float(cnn_confidence),
            "emotion": emotion,
            "is_yawning": is_yawning,
            "face_rects": [{
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }],
            "landmarks": {
                "left_eye": [{'x': int(p[0]), 'y': int(p[1])} for p in left_eye],
                "right_eye": [{'x': int(p[0]), 'y': int(p[1])} for p in right_eye],
                "mouth": [{'x': int(p[0]), 'y': int(p[1])} for p in mouth],
                "all": all_landmarks
            }
        }
    
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return {
            "drowsy": False,
            "ear": 0.0,
            "mar": 0.0,
            "cnn_confidence": 0.0,
            "emotion": "unknown",
            "is_yawning": False,
            "message": f"Error processing image: {str(e)}",
            "face_rects": [],
            "landmarks": {
                "left_eye": [],
                "right_eye": [],
                "mouth": [],
                "all": []
            }
        }

# Pydantic model for alert requests
class AlertRequest(BaseModel):
    user_id: str
    alert_type: str = "drowsiness"
    message: str = "Driver drowsiness detected"
    location: dict = None

# API endpoints
@app.get("/")
async def root():
    return {"message": "Drowsiness Detection API is running"}

@app.get("/health")
async def health_check():
    models_loaded = cnn_model is not None and rf_model is not None and detector is not None and predictor is not None
    return {
        "status": "healthy",
        "models_loaded": models_loaded
    }

@app.post("/api/detect/drowsiness")
async def detect_drowsiness(file: UploadFile = File(...)):
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process image
        result = process_image(img)
        # Ensure all values are JSON serializable
        sanitized_result = {k: sanitize(v) for k, v in result.items()}
        return JSONResponse(content=sanitized_result)
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/alert")
async def send_alert(request: AlertRequest):
    # This would typically connect to a notification service
    # For now, we'll just log the alert and return success
    logger.info(f"Alert received: {request.dict()}")
    return {"success": True, "message": "Alert sent successfully"}

# WebSocket for real-time frame processing
@app.websocket("/ws/drowsiness")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive base64 encoded image
            data = await websocket.receive_text()
            
            try:
                # Decode base64 image
                img_data = base64.b64decode(data.split(',')[1] if ',' in data else data)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    await websocket.send_json({"error": "Invalid image format"})
                    continue
                
                # Process image
                result = process_image(img)
                # Ensure all values are JSON serializable
                sanitized_result = {k: sanitize(v) for k, v in result.items()}
                await websocket.send_json(sanitized_result)
            
            except Exception as e:
                logger.error(f"WebSocket processing error: {str(e)}")
                await websocket.send_json({"error": str(e)})
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run("drowsiness_api:app", host="0.0.0.0", port=8000, reload=True)