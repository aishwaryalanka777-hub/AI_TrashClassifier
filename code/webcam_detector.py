import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("trash_model.h5")

classes = ['cardboard','glass','metal','paper','plastic','trash','organic']
# ⚠️ Make sure this order matches your dataset folders!

# Start webcam
cap = cv2.VideoCapture(0)

print("Webcam started... Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize for model
    img = cv2.resize(frame, (64,64))
    img_array = np.array(img) / 255
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    label = classes[np.argmax(prediction)]

    # Put prediction text on screen
    cv2.putText(frame, label, (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("AI Trash Detector", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
