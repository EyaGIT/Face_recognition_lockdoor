# import cv2
# import numpy as np
# from os import listdir
# from os.path import isfile, join
# import serial
# import time
# import pyttsx3

# # Load the pre-trained face detection model
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Function to speak
# def speak(audio):
#     engine.say(audio)
#     engine.runAndWait()

# # Initialize the text-to-speech engine
# engine = pyttsx3.init('sapi5')
# voices = engine.getProperty('voices')
# engine.setProperty("voice", voices[0].id)
# engine.setProperty("rate", 140)
# engine.setProperty("volume", 1000)

# # Initialize variables
# q = 1
# x = 0
# c = 0
# m = 0
# d = 0

# # Loop to train the face recognition model
# while q <= 2:
#     data_path = 'C:/Users/ASUS/Desktop/microoooooooo/image'
#     onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
#     Training_data, Labels = [], []
#     for i, file in enumerate(onlyfiles):
#         image_path = join(data_path, file)
#         images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if images is None:
#             print(f"Failed to load image: {image_path}")
#             continue
#         Training_data.append(np.asarray(images, dtype=np.uint8))
#         Labels.append(i)

#     Labels = np.asarray(Labels, dtype=np.int32)
#     if len(Training_data) == 0:
#         print("No training data loaded. Check file paths and image formats.")
#         break

#     model = cv2.face.LBPHFaceRecognizer_create()
#     model.train(np.asarray(Training_data), np.asarray(Labels))
#     print("Training complete")
#     q += 1

# # Function to detect faces
# def face_detector(img, size=0.5):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#     if faces is ():
#         return img, []
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
#         roi = img[y:y + h, x:x + w]
#         roi = cv2.resize(roi, (200, 200))

#     return img, roi

# # Initialize the camera
# cap = cv2.VideoCapture(0)

# # Main loop for face recognition
# while True:
#     ret, frame = cap.read()

#     image, face = face_detector(frame)

#     try:
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         result = model.predict(face)
#         if result[1] < 500:
#             confidence = int((1 - (result[1]) / 300) * 100)
#             display_string = str(confidence)
#             cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))

#         if confidence >= 83:
#             cv2.putText(image, "unlocked", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
#             cv2.imshow('face', image)
#             x += 1
#         else:
#             cv2.putText(image, "locked", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
#             cv2.imshow('face', image)
#             c += 1
#     except:
#         cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
#         cv2.imshow('face', image)
#         d += 1
#         pass

#     # Break the loop if necessary
#     if cv2.waitKey(1) == 13 or x == 10 or c == 30 or d == 20:
#         break

# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()

# # Actions based on face recognition results
# if x >= 5:
#     m = 1
#     ard = serial.Serial('com4', 9600)
#     time.sleep(2)
#     var = 'a'
#     c = var.encode()
#     speak("Face recognition complete. It is matching with the database. Welcome, sir. The door is opening for 5 seconds.")
#     ard.write(c)
#     time.sleep(4)
# elif c == 30:
#     speak("Face is not matching. Please try again.")
# elif d == 20:
#     speak("Face is not found. Please try again.")

# if m == 1:
#     speak("Door is closing.")


# import cv2
# import numpy as np
# from os import listdir
# from os.path import isfile, join
# import serial
# import time
# import pyttsx3

# # Load the pre-trained face detection model
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Function to speak
# def speak(audio):
#     engine.say(audio)
#     engine.runAndWait()

# # Initialize the text-to-speech engine
# engine = pyttsx3.init('sapi5')
# voices = engine.getProperty('voices')
# engine.setProperty("voice", voices[0].id)
# engine.setProperty("rate", 140)
# engine.setProperty("volume", 1000)

# # Initialize variables
# q = 1
# x = 0
# c = 0
# m = 0
# d = 0

# # Loop to train the face recognition model
# while q <= 2:
#     data_path = 'C:/Users/ASUS/Desktop/microoooooooo/image'
#     onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
#     Training_data, Labels = [], []
#     for i, file in enumerate(onlyfiles):
#         image_path = join(data_path, file)
#         images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if images is None:
#             print(f"Failed to load image: {image_path}")
#             continue
#         Training_data.append(np.asarray(images, dtype=np.uint8))
#         Labels.append(i)

#     Labels = np.asarray(Labels, dtype=np.int32)
#     if len(Training_data) == 0:
#         print("No training data loaded. Check file paths and image formats.")
#         break

#     model = cv2.face.LBPHFaceRecognizer_create()
#     model.train(np.asarray(Training_data), np.asarray(Labels))
#     print("Training complete")
#     q += 1

# # Function to detect faces
# def face_detector(img, size=0.5):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#     if faces is ():
#         return img, []
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
#         roi = img[y:y + h, x:x + w]
#         roi = cv2.resize(roi, (200, 200))

#     return img, roi

# # Initialize the camera
# cap = cv2.VideoCapture(0)

# # Main loop for face recognition
# while True:
#     ret, frame = cap.read()

#     image, face = face_detector(frame)

#     try:
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         result = model.predict(face)
#         if result[1] < 500:
#             confidence = int((1 - (result[1]) / 300) * 100)
#             display_string = str(confidence)
#             cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))

#         if confidence >= 81:
#             cv2.putText(image, "unlocked", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
#             cv2.imshow('face', image)
#             x += 1
#             # Send result to Arduino
#             print("Sending 'unlocked' to Arduino")
#             # Replace 'com5' with your correct serial port
#             ard = serial.Serial('com6', 9600)
#             time.sleep(2)
#             var = 'a'
#             c = var.encode()
#             ard.write(c)
#             time.sleep(4)
#         else:
#             cv2.putText(image, "locked", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
#             cv2.imshow('face', image)
#             c += 1
#     except:
#         cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
#         cv2.imshow('face', image)
#         d += 1
#         pass

#     # Break the loop if necessary
#     if cv2.waitKey(1) == 13 or x == 10 or c == 30 or d == 20:
#         break

# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()

# # Actions based on face recognition results
# if x >= 5:
#     m = 1
#     if m == 1:
#         speak("Door is closing.")
# elif c == 30:
#     speak("Face is not matching. Please try again.")
# elif d == 20:
#     speak("Face is not found. Please try again.")

import cv2
import numpy as np
import os
import time
import pyttsx3
import serial  # Import the serial module for Arduino communication

# Configuration
DATA_PATH = 'C:/Users/ASUS/Desktop/microoooooooo/image'
MODEL_TRAIN_ITERATIONS = 2
MAX_IMAGES_PER_PERSON = 10
MIN_CONFIDENCE_THRESHOLD = 81
SERIAL_PORT = 'com4'

# Load pre-trained face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 140)
engine.setProperty("volume", 1000)

# Function to speak
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Function to train face recognition model
def train_model(data_path):
    training_data, labels = [], []
    for i, file in enumerate(os.listdir(data_path)):
        image_path = os.path.join(data_path, file)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if images is None:
            print(f"Failed to load image: {image_path}")
            continue
        training_data.append(np.asarray(images, dtype=np.uint8))
        labels.append(i)

    labels = np.asarray(labels, dtype=np.int32)
    if len(training_data) == 0:
        print("No training data loaded. Check file paths and image formats.")
        return None
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(training_data), np.asarray(labels))
    print("Training complete")
    return model

# Function to detect faces
def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        cv2.putText(img, "Face not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        cv2.putText(img, "Face found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img, roi

# Initialize the camera
cap = cv2.VideoCapture(0)

# Train the face recognition model
model = None
for _ in range(MODEL_TRAIN_ITERATIONS):
    model = train_model(DATA_PATH)

if model is None:
    exit()

# Main loop for face recognition
x, c, d = 0, 0, 0
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        if result[1] < 500:
            confidence = int((1 - (result[1]) / 300) * 100)
            display_string = str(confidence)
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))

        if confidence >= MIN_CONFIDENCE_THRESHOLD:
            cv2.putText(image, "unlocked", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
            cv2.imshow('face', image)
            x += 1
            # Send result to Arduino
            print("Sending 'unlocked' to Arduino")
            try:
                ard = serial.Serial(SERIAL_PORT, 9600)
                time.sleep(2)
                var = 'a'
                c = var.encode()
                ard.write(c)
                time.sleep(4)
            except serial.SerialException:
                print("Error: Failed to communicate with Arduino")
        else:
            cv2.putText(image, "locked", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
            cv2.imshow('face', image)
            c += 1
            if c >= 5:
                speak("Face not recognized. Please adjust your position and try again.")
                c = 0
    except:
        cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
        cv2.imshow('face', image)
        d += 1
        pass

    # Break the loop if necessary
    if cv2.waitKey(1) == 13 or x == MAX_IMAGES_PER_PERSON or d == 20:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Actions based on face recognition results
if x >= 5:
    speak("Door is closing.")
elif d == 20:
    speak("Face is not found. Please try again.")
