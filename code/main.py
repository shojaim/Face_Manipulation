import cv2
import numpy as np
from face_landmarks import detect_landmarks
from utils import mouse_callback, set_landmarks, dragging, selected_part

# Global variables for dragging
dragging = False
selected_part = None
x_init, y_init = 0, 0

# Path to image and shape predictor
image_path = '../images/image.jpg'
shape_predictor_path = '../code/utils/shape_predictor_68_face_landmarks.dat'

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise Exception("Image not found")

# Clone image for output
output_image = image.copy()

# Set up windows and mouse callback
cv2.namedWindow('Landmark Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Landmark Window', 720, 720)

cv2.namedWindow('Output Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output Window', 720, 720)

while True:
    # Clone image for displaying landmarks
    display_image = output_image.copy()

    # Detect landmarks
    landmarks = detect_landmarks(display_image, shape_predictor_path)
    if landmarks:
        set_landmarks(landmarks)  # Pass landmarks to the mouse callback function
        param = {'image': output_image, 'output_image': output_image, 'landmarks': landmarks}
        cv2.setMouseCallback('Landmark Window', mouse_callback, param)

        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(display_image, (x, y), 2, (0, 255, 0), -1)

    # Display the images
    cv2.imshow('Landmark Window', display_image)
    cv2.imshow('Output Window', output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
