import cv2
import numpy as np

def main():
    capture = cv2.VideoCapture(0)
    numframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    chunks = numframes // 16
    vid = []
    videoFrames = []
    while (True):
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        cv2.imshow("livestream", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()

main()