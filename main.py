import cv2
import numpy as np
from c3d_feature_extractor import getFeatureExtractor
def main():
    capture = cv2.VideoCapture(0)
    numframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    chunks = numframes // 16
    vid = []
    videoFrames = []
    features = []
    featureExtractor = getFeatureExtractor()
    while (True):
        ret, frame = capture.read()
        if not ret:
            break
        cv2.imshow("livestream", frame)
        frame = cv2.resize(frame, (112, 112))
        for i in range(chunks):
            X = vid[i*16:i*16+16]
            out = featureExtractor.predict(np.array([X]))
            out = out.reshape(4096)
            features.append(out)
            print(out)
        if cv2.waitKey(1) == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()

main()