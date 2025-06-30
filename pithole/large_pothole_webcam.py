import cv2
from ultralytics import YOLO
import csv
import datetime
import os

# --- Configuration ---
# Path to your trained model weights
MODEL_PATH = 'kaggle&RDD.pt' # Ensure this is the correct path to your downloaded model
# Confidence threshold for detections (0.02 is very low, might get many false positives)
CONF_THRESHOLD = 0.156 # Recommend starting higher, like 0.25 or 0.5
# Path for the CSV log file
CSV_LOG_FILE = 'pothole_detections.csv'

def detect_potholes_webcam():
    """
    Detects potholes in real-time using the laptop's webcam and logs detections to CSV.
    """
    try:
        # Load the trained YOLOv8 model
        model = YOLO(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")

        # Open the default webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam. Make sure it's connected and not in use by another application.")
            return

        print("Webcam opened successfully. Press 'q' to quit.")

        # --- CSV Setup ---
        csv_file_exists = os.path.isfile(CSV_LOG_FILE)
        csv_file = open(CSV_LOG_FILE, 'a', newline='') # Open in append mode
        csv_writer = csv.writer(csv_file)

        # Write header only if file is new
        if not csv_file_exists:
            csv_writer.writerow(['Timestamp', 'Latitude', 'Longitude', 'Pothole_ID', 'Class', 'Confidence', 'X_center', 'Y_center', 'Width', 'Height', 'Xmin', 'Ymin', 'Xmax', 'Ymax'])
            print(f"Created new CSV log file: {CSV_LOG_FILE}")
        else:
            print(f"Appending to existing CSV log file: {CSV_LOG_FILE}")

        pothole_counter = 0 # Simple counter for unique pothole ID in this session

        while True:
            ret, frame = cap.read() # Read a frame from the webcam
            if not ret:
                print("Failed to grab frame, exiting...")
                break

            current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Placeholder for actual GPS data
            # If you integrate a GPS module, this is where you'd read its data
            current_latitude = "N/A"
            current_longitude = "N/A"

            # Perform inference on the frame
            results = model(frame, conf=CONF_THRESHOLD)

            # Process detections
            for r in results: # There's usually only one result object per image
                boxes = r.boxes.xywhn.cpu().numpy() # normalized x,y,w,h
                confs = r.boxes.conf.cpu().numpy()  # confidence scores
                clss = r.boxes.cls.cpu().numpy()    # class IDs
                xyxy = r.boxes.xyxy.cpu().numpy()   # absolute pixel xmin,ymin,xmax,ymax

                for i in range(len(boxes)):
                    x_center, y_center, width, height = boxes[i]
                    confidence = confs[i]
                    class_id = int(clss[i])
                    xmin, ymin, xmax, ymax = xyxy[i]

                    # Assuming 'D40' is class 0 (pothole) in your data.yaml
                    # Adjust if your data.yaml maps 'D40' to a different ID
                    if model.names[class_id] == 'D40': # Check if the detected class is 'D40'
                        pothole_counter += 1
                        csv_writer.writerow([
                            current_timestamp,
                            current_latitude,
                            current_longitude,
                            f'POTHOLE_{pothole_counter}', # A unique ID for this session
                            model.names[class_id],
                            f"{confidence:.4f}",
                            f"{x_center:.6f}", f"{y_center:.6f}", f"{width:.6f}", f"{height:.6f}",
                            f"{xmin:.1f}", f"{ymin:.1f}", f"{xmax:.1f}", f"{ymax:.1f}"
                        ])
                        # Ensure data is written immediately
                        csv_file.flush()


            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow('YOLOv8 Pothole Detection (Webcam)', annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release the webcam and destroy all OpenCV windows
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        if 'csv_file' in locals() and not csv_file.closed:
            csv_file.close() # Close the CSV file
            print(f"CSV log file '{CSV_LOG_FILE}' closed.")
        print("Webcam released and windows closed.")

if __name__ == '__main__':
    detect_potholes_webcam()