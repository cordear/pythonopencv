import cv2
import sys
import os
import logging


def skinDetect(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    crFrame = cv2.split(frame)[1]
    crFrame = cv2.GaussianBlur(crFrame, (5, 5), 0)
    _, crFrame = cv2.threshold(
        crFrame, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return crFrame


if __name__ == "__main__":
    logFormat: str = "%(asctime)s - %(levelname)s: %(message)s"
    logpath: str = os.path.split(
        os.path.split(os.path.realpath(__file__))[0])[0]
    logging.basicConfig(format=logFormat,
                        level=logging.DEBUG, filename="{0}/log/log".format(logpath))
    print(cv2.__version__)
    # print(logpath)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hull = []
    hullI = []
    video = cv2.VideoCapture(0)
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
            skinFrame, cv2.MORPH_OPEN, element, iterations=6)
        skinFrame = cv2.morphologyEx(
            skinFrame, cv2.MORPH_CLOSE, element, iterations=6)
        skinFrame = cv2.morphologyEx(
            skinFrame, cv2.MORPH_CLOSE, element, iterations=6)
        skinFrame = cv2.morphologyEx(
            skinFrame, cv2.MORPH_OPEN, element, iterations=6)
        contours, _ = cv2.findContours(
            skinFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        skinFrame = cv2.cvtColor(
            skinFrame, cv2.COLOR_GRAY2BGR)  # Extend frame layers
        skin = cv2.bitwise_and(frame, skinFrame)
        cv2.imshow("Origin", frame)
        cv2.imshow("Skin", skin)
        for t in contours:
            if cv2.contourArea(t) < 100:
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

        frame = cv2.drawContours(frame, hull, -1, (255, 0, 0))
        cv2.imshow("Final", frame)
        hull.clear()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("Quit")
            break
    video.release()
    cv2.destroyAllWindows()
