import cv2
import sys
import os
import logging
if __name__ == "__main__":
    logFormat: str = "%(asctime)s - %(levelname)s: %(message)s"
    logpath: str = os.path.split(
        os.path.split(os.path.realpath(__file__))[0])[0]
    logging.basicConfig(format=logFormat,
                        level=logging.DEBUG, filename="{0}/log/log".format(logpath))
    print(cv2.__version__)
    # print(logpath)
    video = cv2.VideoCapture(0)
    logging.info("Camera set for 0.")
    if(video.isOpened() != True):
        print("Camera open failed.")
        logging.error("Camera open failed.")
        sys.exit()
    while(True):
        ret, frame = video.read()
        frame = cv2.flip(frame, 0)  # Flip the picture. Change if needed
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("Quit")
            break
    video.release()
    cv2.destroyAllWindows()
