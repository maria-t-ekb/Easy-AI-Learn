import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import librosa
import sounddevice as sd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import shutil
from ultralytics import YOLO
import cv2
from pathlib import Path

def y_train(path, y=3, d=0):
    model = YOLO("yolo11s-cls.pt")
    z = path
    y = y
    d = d
    if d == 1:
        d = 'cpu'

    try:
        results = model.train(data=z, epochs=y, imgsz=100, device=d)
        return 'Модель успешно обучена!'    # Проверьте папку runs/classify/train/weights/best.pt'
    except Exception as e:
        return f'Ошибка при обучении: {e}'

def y_test(path):
    z = path
    model = YOLO("best.pt")
    results = model(z)

    if results:
        result = results[0]
        names, probs = process_classification_result(result)
        if names and probs:
            # print("Результаты классификации:")
            top_idx = np.argmax(probs)
            # print(f"\nНаиболее вероятно: {names[top_idx]} ({probs[top_idx]:.2f})")
            return f'Наиболее вероятно: {names[top_idx]} ({probs[top_idx]:.2f})'
