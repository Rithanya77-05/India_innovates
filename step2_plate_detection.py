from ultralytics import YOLO
import cv2


# Load plate detection model
model = YOLO("license_plate.pt")

# Read image
image = cv2.imread("image.jpeg")

# Run YOLO detection
results = model(image)

for r in results:
    boxes = r.boxes.xyxy

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

        # Label
        cv2.putText(image, "Plate", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Crop plate
        plate = image[y1:y2, x1:x2]

        cv2.imshow("Detected Plate", plate)

# Show final image
cv2.imshow("Plate Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
