from fastapi import FastAPI, HTTPException, Body, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import io
from ultralytics import YOLO
import uvicorn
import os
import json
import asyncio
from typing import List, Dict, Any, Optional

# Initialize FastAPI app
app = FastAPI(title="Pothole Detection API")

# Add CORS middleware to allow requests from mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Path to your trained model weights
MODEL_PATH = 'kaggle&RDD.pt'  # Make sure this path is correct

# Load the YOLO model
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. API will return mock data.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define request model
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image
    location: Optional[Dict[str, float]] = None

# Define response model
class PotholeResponse(BaseModel):
    potholes: List[Dict[str, Any]]
    
# Define location model
class Location(BaseModel):
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    speed: Optional[float] = None
    accuracy: Optional[float] = None

@app.get("/")
async def root():
    return {"message": "Pothole Detection API is running"}

@app.post("/detect_pothole", response_model=PotholeResponse)
async def detect_pothole(request: Dict[str, Any] = Body(...)):
    try:
        # Get base64 encoded image from request
        base64_image = request.get("image")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Get location data if available
        location_data = request.get("location")
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_image)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process the image and detect potholes
        potholes = await process_image_for_potholes(image)
        
        # Add location data to each pothole if available
        if location_data:
            for pothole in potholes:
                pothole.update({
                    "latitude": location_data.get("latitude"),
                    "longitude": location_data.get("longitude"),
                    "altitude": location_data.get("altitude"),
                    "speed": location_data.get("speed"),
                    "accuracy": location_data.get("accuracy")
                })
        
        return {"potholes": potholes}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


async def process_image_for_potholes(image):
    """Process an image and detect potholes"""
    if model is not None:
        # Lower confidence threshold to detect more potholes
        results = model(image, conf=0.10)  # Reduced from 0.25 to detect more potholes
        
        potholes = []
        for r in results:  # Usually only one result per image
            boxes = r.boxes.xywhn.cpu().numpy()  # normalized x,y,w,h
            confs = r.boxes.conf.cpu().numpy()   # confidence scores
            clss = r.boxes.cls.cpu().numpy()     # class IDs
            xyxy = r.boxes.xyxy.cpu().numpy()    # absolute pixel xmin,ymin,xmax,ymax
            
            for i in range(len(boxes)):
                x_center, y_center, width, height = boxes[i]
                confidence = float(confs[i])
                class_id = int(clss[i])
                xmin, ymin, xmax, ymax = map(float, xyxy[i])
                
                # Calculate dimensions in cm (this is an approximation)
                # In a real app, you would need camera calibration for accurate measurements
                width_cm = float(width * 100)  # Arbitrary scaling for demonstration
                height_cm = float(height * 100)
                
                # Add detection to results
                potholes.append({
                    "confidence": float(confidence),
                    "class": model.names[class_id],
                    "width": float(width_cm),
                    "height": float(height_cm),
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "xmin": float(xmin),
                    "ymin": float(ymin),
                    "xmax": float(xmax),
                    "ymax": float(ymax)
                })
    else:
        # If model is not available, return improved mock data with better visualization
        # Get image dimensions to create more realistic bounding boxes
        height, width = image.shape[:2] if image is not None else (720, 1280)
        
        # Calculate center position - place pothole in center of image
        center_x = width / 2
        center_y = height / 2
        
        # Calculate box size - make it about 20% of image size for visibility
        box_width = width * 0.2
        box_height = height * 0.2
        
        # Calculate box coordinates
        xmin = center_x - (box_width / 2)
        ymin = center_y - (box_height / 2)
        xmax = center_x + (box_width / 2)
        ymax = center_y + (box_height / 2)
        
        print(f"Creating mock pothole with dimensions: {width}x{height}, box: ({xmin},{ymin},{xmax},{ymax})")
        
        potholes = [{
            "confidence": 0.95,  # Higher confidence for better visibility
            "class": "D40",
            "width": 45.5,
            "height": 30.2,
            "x_center": center_x / width,  # Normalized coordinates
            "y_center": center_y / height,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        }]
    
    return potholes


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is not None:
        return {"status": "ok", "message": "Model loaded successfully"}
    else:
        return {"status": "warning", "message": "Model not loaded, returning mock data"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            if data_json.get("type") == "frame":
                # Get base64 encoded image
                base64_image = data_json.get("image")
                if not base64_image:
                    await websocket.send_json({"error": "No image provided"})
                    continue
                
                # Get location data if available
                location_data = data_json.get("location")
                
                # Decode base64 image
                try:
                    image_bytes = base64.b64decode(base64_image)
                    image_array = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        await websocket.send_json({"error": "Invalid image format"})
                        continue
                    
                    # Process the image and detect potholes
                    potholes = await process_image_for_potholes(image)
                    
                    # Add location data to each pothole if available
                    if location_data:
                        for pothole in potholes:
                            pothole.update({
                                "latitude": location_data.get("latitude"),
                                "longitude": location_data.get("longitude")
                            })
                    
                    # Send detection results back to client
                    if potholes:
                        await websocket.send_json({
                            "type": "pothole_detection",
                            "potholes": potholes
                        })
                except Exception as e:
                    await websocket.send_json({"error": f"Error processing image: {str(e)}"})
            else:
                await websocket.send_json({"error": "Unknown message type"})
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Clean up when connection is closed
        print("WebSocket connection closed")


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run("pothole_api:app", host="0.0.0.0", port=8000, reload=True)