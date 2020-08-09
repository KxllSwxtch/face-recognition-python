import cv2
from random import randrange

cv2.namedWindow('Face Recognition using Python')

# Import testing images
firstImage = cv2.imread('./assets/first_test.jpg')
secondImage = cv2.imread('./assets/second_test.jpg')

# Setup the testing data file
faceRecognitionData = './face_data.xml'
faceRecognition = cv2.CascadeClassifier(faceRecognitionData)

# Get the access to the video camera
videoCapture = cv2.VideoCapture(0)

if videoCapture.isOpened():
    rval, frame = videoCapture.read()
else:
    rval = False

while rval:
    # Get the video source coming from camera
    rval, frame = videoCapture.read()
    # Process the input frames
    frameGrayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = faceRecognition.detectMultiScale(frameGrayScale)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 2)
    cv2.imshow('Face preview', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


videoCapture.release()
cv2.destroyAllWindows("Face")

# Transform the testing images into GRAYSCALE
cv2.imshow('Face Recognition Program using Python', firstImage)

# Wait until the key is pressed
cv2.waitKey()
