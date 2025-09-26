# Import required libraries
import torch          # PyTorch for loading YOLOv5 model
import cv2            # OpenCV for image & video processing
import numpy as np    # NumPy for array manipulation

# Load the pre-trained YOLOv5 model
# "yolov5s" = small version (fast, lightweight, less accurate)
# pretrained=True downloads weights trained on COCO dataset (80 object classes)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


# Function: Detect objects in a single image
def detect_objects(image_path):
    # Read the image from the given file path
    image = cv2.imread(image_path)

    # Run the YOLOv5 model on the image
    results = model(image)

    # Display results in a new window (with bounding boxes, labels, confidence)
    results.show()


# Function: Real-time object detection using webcam
def detect_from_webcam():
    # Start capturing video from webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Read each frame from webcam
        ret, frame = cap.read()

        # Run YOLOv5 model on the current frame
        results = model(frame)

        # Render results (bounding boxes & labels) on the frame
        # results.render() returns a list of images with drawings → we take the first one
        rendered_img = np.array(results.render()[0])

        # Show the processed frame in a window
        cv2.imshow("Real-Time Object Detection", rendered_img)

        # Press "q" to quit the webcam stream
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


# Main Program – User chooses mode
print("Choose an option:\n1. Detect objects in an image\n2. Real-time object detection from webcam")
choice = input("Enter 1 or 2: ")

if choice == "1":
    # If user chooses image detection
    image_path = input("Enter image path: ")
    detect_objects(image_path)  # Call image detection function

elif choice == "2":
    # If user chooses real-time webcam detection
    detect_from_webcam()

else:
    # Handle invalid choice
    print("Invalid choice. Exiting.")


