import cv2
from ultralytics import YOLO

#model = YOLO("best.pt")
#model = YOLO("yolo11n.pt")
model = YOLO("best.onnx")
#model = YOLO("best.engine")
#model = YOLO("final.onnx")
#model = YOLO("final.engine")

video_path = "mapping9.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)
foundobjects = {}
relaydrop = 0
while True:
    success, frame = cap.read()
    if not success:
        break  # Break the loop if the video ends

    # Perform inference on the frame
    results = model(frame)

    # Visualize predictions (optional)
    annotated_frame = results[0].plot()  # Annotate the frame with detections

    for result in results[0].boxes:
        xcenter, ycenter, width, height = result.xywh[0].tolist()
        xcenter=round(xcenter)
        ycenter=round(ycenter)
        print(f"{xcenter} x {ycenter}")
        cv2.circle(annotated_frame, (xcenter, ycenter), radius=20, color=(0,0,255), thickness=5)
        conf = result.conf[0].item()
        class_index = result.cls[0].item()
        class_name = results[0].names[int(class_index)]
        print(class_name)
        print(relaydrop)
        if relaydrop < 1 and conf > 0.6:
            print("Object Found")
            if class_name in foundobjects:
                foundobjects[class_name] = foundobjects.get(class_name) + 1
                print("Payload Dropped")
                relaydrop+=1
            else:
                foundobjects[class_name] = 1

    # Display the annotated frame
    cv2.imshow("YOLOv11 Object Detection", annotated_frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
