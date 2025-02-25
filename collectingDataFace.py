# import cv2
# # Load the Haar cascade classifier for face detection
# face_classifier = cv2.CascadeClassifier('C:/Users/ASUS/Desktop/microoooooooo/package/haarcascade_frontalface_default.xml')

# def face_extractor(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Detect faces in the grayscale image
#     faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#     if len(faces) == 0:
#         # No faces detected
#         return None
#     # Extract the first detected face
#     (x, y, w, h) = faces[0]
#     cropped_face = img[y:y+h, x:x+w]
#     return cropped_face

# cap = cv2.VideoCapture(0)
# count = 0
# while True:
#     ret, frame = cap.read()
#     if face_extractor(frame) is not None:
#         count += 1
#         face = cv2.resize(face_extractor(frame), (200, 200))
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         file_name_path = 'C:/Users/ASUS/Desktop/microoooooooo/image/' + str(count) + '.jpg'
#         cv2.imwrite(file_name_path, face)
#         cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#         cv2.imshow("Face Cropper", face)
#     else:
#         print('Face not found')
#     if cv2.waitKey(1) == 13 or count == 500:
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Collecting samples complete")

import cv2
import os

# Load the Haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        # No faces detected
        return None
    # Extract all detected faces
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        cropped_faces.append(cv2.resize(cropped_face, (200, 200)))
    return cropped_faces

cap = cv2.VideoCapture(0)
output_dir = 'C:/Users/ASUS/Desktop/microoooooooo/image/'  # Change this to your desired output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count = 0
max_images_per_person = 100  # Set the maximum number of images to collect per person
current_person = 1
while True:
    ret, frame = cap.read()
    faces = face_extractor(frame)
    if faces is not None:
        for face in faces:
            count += 1
            file_name = f'person_{current_person}_{count}.jpg'
            file_path = os.path.join(output_dir, file_name)
            cv2.imwrite(file_path, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Cropper", face)
            if count >= max_images_per_person:
                count = 0
                current_person += 1
                break
    else:
        print('Face not found')
    if cv2.waitKey(1) == 13 or current_person > 2:
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting samples complete")
