import time
import logging

"""Flask Modules for API"""
import flask
from flask import Flask, jsonify, request
"""Deep Learning Modules"""
import cv2
import numpy as np
from keras.models import Model
from c3d_feature_extractor import getFeatureExtractor
import multiprocessing

def init_model():
    try:
        feature_extractor = getFeatureExtractor("weights/weights.h5", "feature_extractor.h5", "fc6", False)
        logging.info("Feature Extraction Model Loaded")
        return feature_extractor
    except Exception as e:
        logging.error(f"Error Loading Feature Extraction Model Message:" +e.value)
        return None
def preprocess_frames(frames):
    for frame in frames:
        cv2.resize(frame, (112, 112))
    return frames
def camera_fps(capture):
    return int(capture.get(cv2.CAP_PROP_FPS))

"""Initiate Flask API"""
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict(feature_extractor,frames):
    return jsonify(result="Violence Detected")
    pass

@app.route('/', methods=['GET'])
def test():
    return jsonify(message="Hello World")


def main():
    print("Initializing Camera...")
    capture = cv2.VideoCapture(0)
    #fps = int(capture.get(cv2.CAP_PROP_FPS))
    #print(f"Frames per second: {fps}")
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
    app.run(debug=True, host='0.0.0.0')
    #main()