import cv2
import os

active = True
path = "faces/"

print("Starter koden...")
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("archive_ansiktkjennetegn/cascades/data/haarcascade_frontalface_alt2.xml")

while active:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:
        print("x: %d\ty: %d\tw: %d\th: %d" % (x,y,w,h))

        print("Lager ny identitet...")
        roi_color = frame[y:y+h+20, x:x+w+20]
        creating = input("Skriv inn navn: ")

        filess = os.listdir(path)

        cv2.imwrite(f"faces/{creating}.png", roi_color)
        print("Bildet lagret!")
        active = False