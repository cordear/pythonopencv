import cv2
import sys
import os
import logging
import numpy as np


def skinDetect(frame):
    Ycframe = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    crFrame = cv2.split(Ycframe)[1]
    crFrame = cv2.GaussianBlur(crFrame, (5, 5), 0)
    _, crFrame = cv2.threshold(
        crFrame, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return crFrame


def display(frame, t, text):
    x, y, w, h = cv2.boundingRect(t)
    cv2.rectangle(frame, (x, y), (x+w, y+h),
                  color=(0, 0, 255), thickness=2)
    cv2.putText(frame, text,
                (x+int(w/2), int(y+h/2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    return frame


if __name__ == "__main__":
    logFormat: str = "%(asctime)s - %(levelname)s: %(message)s"
    logpath: str = os.path.split(
        os.path.split(os.path.realpath(__file__))[0])[0]
    logging.basicConfig(format=logFormat,
                        level=logging.DEBUG, filename="{0}/log/log".format(logpath))
    print("opencv version: "+cv2.__version__)
    # print(logpath)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hull = []
    hullI = []
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    acuteAngle = 0

    logging.info("Camera set for 0.")
    if(video.isOpened() != True):
        print("Camera open failed.")
        logging.error("Camera open failed.")
        sys.exit()
    while(True):
        ret, frame = video.read()
        frame = cv2.flip(frame, 0)  # Flip the picture. Change if needed
        skinFrame = skinDetect(frame)
        skinFrame = cv2.morphologyEx(
            skinFrame, cv2.MORPH_OPEN, element, iterations=3)
        skinFrame = cv2.morphologyEx(
            skinFrame, cv2.MORPH_CLOSE, element, iterations=3)
        skinFrame = cv2.morphologyEx(
            skinFrame, cv2.MORPH_CLOSE, element, iterations=3)
        skinFrame = cv2.morphologyEx(
            skinFrame, cv2.MORPH_OPEN, element, iterations=3)
        contours, _ = cv2.findContours(
            skinFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        skinFrame = cv2.cvtColor(
            skinFrame, cv2.COLOR_GRAY2BGR)  # Extend frame layers
        skin = cv2.bitwise_and(frame, skinFrame)
        cv2.imshow("Origin", frame)
        cv2.imshow("Skin", skin)
        for t in contours:
            if cv2.contourArea(t) < 200:
                continue
            hull.append(cv2.convexHull(t))
            hullI = cv2.convexHull(t, clockwise=True, returnPoints=False)
            defects = cv2.convexityDefects(t, hullI)
            if defects is None:
                continue
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                d = d/256
                if(d <= 40 or d >= 150):
                    continue
                start = t[s][0]
                end = t[e][0]
                far = t[f][0]
                cv2.line(frame, start, far, (0, 255, 0), 2)
                cv2.line(frame, end, far, (0, 255, 0), 2)
                cv2.circle(frame, start, 6, (255, 0, 0))
                cv2.circle(frame, end, 6, (0, 255, 0))
                cv2.circle(frame, far, 6, (0, 0, 255))

                vectorA = [far[0]-start[0], far[1]-start[1]]
                vectorB = [far[0]-end[0], far[1]-end[1]]
                vectorA = np.asarray(vectorA, dtype=np.double)
                vectorB = np.asarray(vectorB, dtype=np.double)
                ds = np.dot(vectorA, vectorB)
                if(ds/(np.linalg.norm(vectorA)*np.linalg.norm(vectorB)) > 0):
                    acuteAngle = acuteAngle+1
            if(acuteAngle < 1 and cv2.arcLength(t, False) > 600):
                frame = display(frame, t, "Rock")
            elif(acuteAngle == 1):
                frame = display(frame, t, "Scissors")
            elif(acuteAngle >= 3):
                frame = display(frame, t, "Paper")
            elif(cv2.arcLength(t, False) > 1000):
                frame = display(frame, t, "Unknow")
            acuteAngle = 0
        frame = cv2.drawContours(frame, hull, -1, (255, 0, 0))

        cv2.imshow("Final", frame)
        hull.clear()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("Quit")
            break
    video.release()
    cv2.destroyAllWindows()
