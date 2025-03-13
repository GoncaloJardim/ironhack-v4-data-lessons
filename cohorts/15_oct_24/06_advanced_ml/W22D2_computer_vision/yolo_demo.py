# pip install yolov5  
import torch
import cv2

# Load the YOLOv3 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference
    results = model(frame)
    
    # Render the results on the frame
    annotated_frame = results.render()[0]
    
    # Display the frame
    cv2.imshow('YOLOv3 Live Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
