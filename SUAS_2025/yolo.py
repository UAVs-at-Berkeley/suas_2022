import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

video_path = "mapping5.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

while True:
    success, frame = cap.read()
    if not success:
        break  # Break the loop if the video ends

    # Perform inference on the frame
    results = model(frame)

    # Visualize predictions (optional)
    annotated_frame = results[0].plot()  # Annotate the frame with detections

    # Display the annotated frame
    cv2.imshow("YOLOv11 Object Detection", annotated_frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()