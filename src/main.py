import cv2
import sys

if __name__ == "__main__":
    print(cv2.__version__)
    video = cv2.VideoCapture(0)

    if(video.isOpened() != True):
        print("Camera open failed.")
        sys.exit()
    while(True):
        ret, frame = video.read()
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()
