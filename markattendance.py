import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyrebase

firebaseConfig = {'apiKey': "AIzaSyA1GfySLSMBlz_2fHbEILa2x9sDkTdy_Cs",
                  'authDomain': "markattedance.firebaseapp.com",
                  'projectId': "markattedance",
                  'storageBucket': "markattedance.appspot.com",
                  'messagingSenderId': "600305140590",
                  'appId': "1:600305140590:web:69bdfa1dc6ebeb0fb8f77f",
                  'measurementId': "G-LSFZ1P1KZ6",
                  'databaseURL': "https://markattedance-default-rtdb.firebaseio.com/"
                  }

firebase = pyrebase.initialize_app(firebaseConfig)

db = firebase.database()

# from PIL import ImageGrab

path = 'img'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        encodeList = []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    db.child(name).set(dtString)

    #db.push(name)
    #db.push(dtString)

    # wi#th open('Attendance.csv', 'r+') as f:myDataList = f.readlines()
    # nameList = []
    # for line in myDataList:
    # entry = line.split(',')
    # nameList.append(entry[0])

    # if name not in nameList:
    # now = datetime.now()
    # dtString = now.strftime('%H:%M:%S')

    # f.writelines(f'n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)
