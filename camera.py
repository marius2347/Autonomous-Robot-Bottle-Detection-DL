# bottle detector using camera and opencv
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# load the best model
best_model = load_model('./models/best_model.h5')

# open the camera index 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Eroare la deschiderea camerei!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nu s-a putut citi frame-ul!")
        break

    # BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # resize to 224x224
    img_resized = cv2.resize(img_rgb, (224, 224))
    # normalization
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # get the prediction
    prediction = best_model.predict(img_array)[0][0]
    label = "no bottle" if prediction > 0.5 else "bottle"
    prob_text = f"Probabilitate: {prediction:.4f}"

    # print the result on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, prob_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the frame
    cv2.imshow("Detectie in timp real", frame)

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
