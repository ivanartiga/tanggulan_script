import time
import cv2
import numpy as np
from keras.models import Model
from c3d_feature_extractor import getFeatureExtractor, predict
import multiprocessing
def main():
    print("Initializing Camera...")
    capture = cv2.VideoCapture(0)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    print(f"Frames per second: {fps}")
    chunks = []
    frame_count = 0
    print("Loading feature extractor model...")
    # feature_extractor = getFeatureExtractor("weights/weights.h5","feature_extractor.h5","fc6",False)
    # prediction = multiprocessing.Process(target=predict(feature_extractor,chunks))
    print("Starting Camera")
    while (True):
        ret, frame = capture.read()
        if not ret:
            break
        cv2.resize(frame, (112, 112))
        cv2.imshow('frame', frame)
        if frame_count < 64:
            chunks.append(frame)
            frame_count = frame_count+1
        else:
            # predict(feature_extractor,chunks[0:16])
            print(len(chunks))
            chunks.clear()
            frame_count = 0
        if cv2.waitKey(1) == ord("q"):
            break
            capture.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()