import cv2
import numpy as np

# Load age and gender models
age_net = cv2.dnn.readNetFromCaffe('C:/Users/Dheeraj/OneDrive/Desktop/New folder (4)/model/age_deploy.prototxt', 'C:/Users/Dheeraj/OneDrive/Desktop/New folder (4)/model/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('C:/Users/Dheeraj/OneDrive/Desktop/New folder (4)/model/gender_deploy.prototxt', 'C:/Users/Dheeraj/OneDrive/Desktop/New folder (4)/model/gender_net.caffemodel')

# List of age and gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(48-53)','(60-100)']
gender_list = ['Male', 'Female']

# Open video capture
cap = cv2.VideoCapture(0)

# Loop through frames
while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Pass the blob through the network
    age_net.setInput(blob)
    age_preds = age_net.forward()
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()

    # Get predicted age and gender
    age_index = np.argmax(age_preds)
    gender_index = np.argmax(gender_preds)

    # Display age and gender
    age = age_list[age_index]
    gender = gender_list[gender_index]
    label = f"{gender}, {age}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Age and Gender Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()