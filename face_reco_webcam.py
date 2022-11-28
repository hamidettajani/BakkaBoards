import numpy as np
import face_recognition as fr
import cv2
import shutil, os
import time


video_capture = cv2.VideoCapture(0) # Starter video.

jacky_bilder = fr.load_image_file("faces/11musk.png") # laste bilde fil. Brukes til kjennetegn
jacky_ansiktencode = fr.face_encodings(jacky_bilder)[0] # encoding. Som at denne linjen tar bildet og analyserer. avstand mellom øynene, nese, størrelse på munnen osv. numeration av ansikt. Vi putter 0 fordi denne linjen returnerer arrays. VIl bare ha en

musk_bilder = fr.load_image_file("faces/11musk.png")
musk_ansiktencode = fr.face_encodings(musk_bilder)[0]

henrik_bilder = fr.load_image_file("faces/5henrik.png")
henrik_ansiktencode = fr.face_encodings(henrik_bilder)[0]

ola_bilder = fr.load_image_file("faces/12ola.png")
ola_ansiktencode = fr.face_encodings(ola_bilder)[0]

morten_bilder = fr.load_image_file("faces/10morten.png")
morten_ansiktencode = fr.face_encodings(morten_bilder)[0]

isaac_bilder = fr.load_image_file("faces/6isaac.png")
isaac_ansiktencode = fr.face_encodings(isaac_bilder)[0]

diego_bilder = fr.load_image_file("faces/4diego.png")
diego_ansiktencode = fr.face_encodings(diego_bilder)[0]

amir_bilder = fr.load_image_file("faces/1amir.png")
amir_ansiktencode = fr.face_encodings(amir_bilder)[0]

magnus_bilder = fr.load_image_file("faces/8magnus_elev.png")
magnus_ansiktencode = fr.face_encodings(magnus_bilder)[0]

david_bilder = fr.load_image_file("faces/3david.png")
david_ansiktencode = fr.face_encodings(david_bilder)[0]

syver_bilder = fr.load_image_file("faces/13syver.png")
syver_ansiktencode = fr.face_encodings(david_bilder)[0]

Magnus_bilder = fr.load_image_file("faces/9Magnus_lærer.png")
Magnus_ansiktencode = fr.face_encodings(Magnus_bilder)[0]

bean_bilder = fr.load_image_file("faces/2bean.png")
bean_ansiktencode = fr.face_encodings(bean_bilder)[0]

# Til å få ansiktet fra front
face_cascade = cv2.CascadeClassifier("archive_ansiktkjennetegn/cascades/data/haarcascade_frontalface_alt2.xml")


# Protocol
active = True
face_unknown = False


kjent_ansikt_encode = [jacky_ansiktencode, musk_ansiktencode, henrik_ansiktencode, ola_ansiktencode,
morten_ansiktencode, isaac_ansiktencode, diego_ansiktencode, amir_ansiktencode, magnus_ansiktencode, david_ansiktencode, syver_ansiktencode, Magnus_ansiktencode, bean_ansiktencode] # Rekkefølge er viktig. Ellers så viser navnet feil på kamera
kjent_ansikt_navn = ["Jacky", "Musky", "Henrik", "Ola", "Morten", "Isaac", "Diego", "Amir", "Magnus P.O", "David", "Syver", "Magnus .T", "Bean"]

while active: # Neste steg. Kjennetegne på video. Bruk av while True. Framme for framme
    ret, frame = video_capture.read() # ret betyr boolean og returnerer true hvis frame er tilgjengelig. (Lurer på hvorfor den brukes ikke. Bør undersøke på internett) : frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) # Endre størrelse på framme
    #rgb_frame = small_frame[:, :, ::-1] # Fanger RGB på frammen. Printer ut RGB.
    #print(frame)

    ansikt_locations = fr.face_locations(small_frame) # Finner hvor er ansikt i framme. Kan være flere ansikter.
    ansikt_encoding = fr.face_encodings(small_frame, ansikt_locations) # tar ramme fra kamera og sjekker fargen. Liksom hvor er ansiktet.

    for (top, right, bottom, left), ansikt_encoding in zip(ansikt_locations, ansikt_encoding): # Location lager array for top right bottom og left. Encoding iterates top, right osv.

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        

        match = fr.compare_faces(kjent_ansikt_encode, ansikt_encoding) # Analysere ansikt. Sammenligne hvis koden kjenner musk eller meg eller hvilken ansikt (kjent_ansikt_encode)

        navn = "Ukjent"

        face_distance = fr.face_distance(kjent_ansikt_encode, ansikt_encoding) # Sammenligne mellom analysert ansikt - normal ansikt

        best_match_index = np.argmin(face_distance) # Hvis koden kjenner ansikt, print ut spesifikk index. # np.argmin returnerer smallest mulig verdi


        if match[best_match_index]: # Kjekker for den beste eller mest lignene ansikt
            navn = kjent_ansikt_navn[best_match_index] # Hvis true, viser det "Elon Musk" eller hva du endrer på linje 11.
            # print(best_match_index)
        else:
            for (x,y,w,h) in faces:
                print("Face Not Recognized. Creating New Identity...")
                roi_color = frame[y:y+h, x:x+w]
                creating = input("Enter Your Name: ")

                path = "faces/"
                filess = os.listdir(path)

                cv2.imwrite(f"{len(filess)}{creating}.png", roi_color)
                face_unknown = True
                active = False



        # if navn == "Jacky":
        #     print("STOOPID CRIME FOUND!")
        #     exit()
            


        # Lage rektangel

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 100, 0), 2) # frame for bildet, mest betyr størrelse på rektangel, farge og tykkhet av rektangel.

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 100, 0), cv2.FILLED) # Lag rektangel under.
        font = cv2.FONT_HERSHEY_SIMPLEX # Velg hvilken font det skal se ut.
        cv2.putText(frame, navn, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2) # Plassering av navn, farger, tykkhet og størrelsen.

        # print("Top:", top, "Right:", right, "Bottom:", bottom, "Left:", left)

    cv2.imshow("Big Boi", frame)

    if cv2.waitKey(1) & 0xFF == ord("d"): # waitKey er sikkert frame per second. Settes på 0, fryser video, 1000 er 1 sek per framme. 1 er den beste jeg kan kjøre FPS kjappere på. For å øke FPS, vet jeg bare å gjøre videoen mindre.
        break


video_capture.release() # Bli ferdig med video fra webcam
cv2.destroyAllWindows() 

time.sleep(2)    

if face_unknown == True:
    path = "faces/.."
    files = os.listdir(path)




    for file in files:
        name, type = os.path.splitext(file)
        type = type[1:] 
        # print(type == "png", "=", type)

        
        if os.path.exists(path + "/" + "faces"): # Hvis faces mappe finnes, put png inn 

            if type != "png":
                continue
            else:
                shutil.move(path + "/" + name + "." + type, path + "/" + "faces")
                    

            
    else:
        print("Done! Moved it to 'faces' directory.")
