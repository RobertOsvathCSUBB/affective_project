# the first one is the model trained on FER2013
from emotion_classifier_fer import emotion_classifier
import cv2
import torch
import torchvision.transforms.v2 as T
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformation = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32),
    T.Lambda(lambda x: x / 255.),
    T.Normalize(mean=[0.5], std=[0.5])
])

#Model Evaluation on test data
classes = ('Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral')

#helper_function for real time testing
def load_img(path):
    img = Image.open(path)
    img = transformation(img)
    img = torch.autograd.Variable(img,requires_grad = True)
    img = img.unsqueeze(0)
    return img.to(device)

if __name__ == "__main__":

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Capture the video
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Draw the bounding boxes
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi,(48,48))
            cv2.imwrite("roi.jpg", roi)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        time.sleep(0.1)

        # Predict the emotion
        img = load_img("roi.jpg")
        output = emotion_classifier(img)
        prediction = torch.argmax(output["probs"])
        predicted_class = classes[prediction.item()]

        # Display the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.putText(frame, predicted_class, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('emotion detection', frame)

        # Stop if (Q) key is pressed
        k = cv2.waitKey(30)
        if k==ord("q"):
            break

    # Release the VideoCapture object
    cap.release()