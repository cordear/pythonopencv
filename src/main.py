import cv2
import sys
import logging
if __name__ == "__main__":
    logFormat: str = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(filename="./venv/log/log", format=logFormat,
                        level=logging.DEBUG)
    print(cv2.__version__)
    print(sys.argv[0])
    video = cv2.VideoCapture(0)
    logging.info("Camera set for 0.")
    if(video.isOpened() != True):
        print("Camera open failed.")
        logging.error("Camera open failed.")
        sys.exit()
    while(True):
        ret, frame = video.read()
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("Quit")
            break
    video.release()
    cv2.destroyAllWindows()
