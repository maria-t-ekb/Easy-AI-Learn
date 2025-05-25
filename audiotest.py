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


def create_dataset():
    dataset_path = input("Введите имя для нового датасета (будет создана папка с этим именем): ")

    if os.path.exists(dataset_path):
        print("Ошибка: папка уже существует!")
        return

    os.makedirs(dataset_path)

    print("Создание нового датасета. Введите 'стоп' для завершения.")

    while True:
        class_name = input("\nВведите имя нового класса (или 'стоп' для завершения): ")
        if class_name.lower() == 'стоп':
            break

        class_path = os.path.join(dataset_path, class_name)
        os.makedirs(class_path, exist_ok=True)

        print(f"\nКласс '{class_name}'. Введите 'далее' для перехода к следующему классу.")

        sample_count = 0
        while True:
            action = input(f"Нажмите Enter для записи образца {sample_count + 1} или введите 'далее': ")
            if action.lower() == 'далее':
                break

            try:
                duration = 3
                sample_rate = 22050
                print(f"Записываю {duration} секунды... (говорите сейчас)")
                audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
                sd.wait()
                audio = np.squeeze(audio)
                filename = os.path.join(class_path, f"sample_{sample_count}.wav")
                sf.write(filename, audio, sample_rate)
                print(f"Образец сохранен как {filename}")

                sample_count += 1
            except Exception as e:
                print(f"Ошибка при записи: {e}")


def load_dataset(dataset_path):
    features = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for audio_file in os.listdir(label_path):
                audio_path = os.path.join(label_path, audio_file)
                try:
                    data, sample_rate = librosa.load(audio_path)
                    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
                    mfccs_processed = np.mean(mfccs.T, axis=0)

                    features.append(mfccs_processed)
                    labels.append(label)
                except Exception as e:
                    print(f"Ошибка при обработке файла {audio_path}: {e}")

    return np.array(features), np.array(labels)


def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def record_audio(duration=3, sample_rate=22050):
    print(f"Записываю {duration} секунды аудио...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    audio = np.squeeze(audio)
    return audio, sample_rate


def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed


def train_new_model():
    dataset_path = input("Введите путь к датасету (папка с папками классов): ")
    if not os.path.exists(dataset_path):
        print("Ошибка: указанный путь не существует!")
        return
    epochs = int(input("Введите количество эпох обучения: "))
    model_name = input("Введите имя для сохранения модели (без расширения): ") + ".h5"
    print("Загрузка датасета...")
    features, labels = load_dataset(dataset_path)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)
    input_shape = (X_train.shape[1],)
    num_classes = len(le.classes_)
    model = create_model(input_shape, num_classes)

    print("Обучение модели...")
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    model.save(model_name)
    np.save(model_name.replace('.h5', '_classes.npy'), le.classes_)

    print(f"Модель сохранена как {model_name}")
    print(f"Классы: {', '.join(le.classes_)}")


def use_model_realtime():
    model_path = input("Введите имя файла модели (.h5): ")
    if not os.path.exists(model_path):
        print("Ошибка: файл модели не существует!")
        return

    classes_path = model_path.replace('.h5', '_classes.npy')
    if not os.path.exists(classes_path):
        print("Ошибка: файл классов не найден!")
        return

    model = tf.keras.models.load_model(model_path)
    classes = np.load(classes_path)

    print("Модель загружена. Готов к классификации звуков.")
    print("Нажмите Ctrl+C для выхода.")

    try:
        while True:
            audio, sample_rate = record_audio()
            features = extract_features(audio, sample_rate)
            features = features.reshape(1, -1)
            predictions = model.predict(features)
            predicted_index = np.argmax(predictions)
            predicted_class = classes[predicted_index]
            confidence = predictions[0][predicted_index]

            print(f"Предсказанный класс: {predicted_class} (вероятность: {confidence:.2f})")
            time.sleep(1)

    except KeyboardInterrupt:
        print("Остановка классификации в реальном времени")


def use_model_on_file():
    model_path = input("Введите имя файла модели (.h5): ")
    if not os.path.exists(model_path):
        print("Ошибка: файл модели не существует!")
        return

    audio_path = input("Введите путь к аудиофайлу для тестирования: ")
    if not os.path.exists(audio_path):
        print("Ошибка: аудиофайл не существует!")
        return

    classes_path = model_path.replace('.h5', '_classes.npy')
    if not os.path.exists(classes_path):
        print("Ошибка: файл классов не найден!")
        return

    model = tf.keras.models.load_model(model_path)
    classes = np.load(classes_path)

    try:
        data, sample_rate = librosa.load(audio_path)
        features = extract_features(data, sample_rate)
        features = features.reshape(1, -1)

        # Предсказываем класс
        predictions = model.predict(features)
        predicted_index = np.argmax(predictions)
        predicted_class = classes[predicted_index]
        confidence = predictions[0][predicted_index]

        print("\nРезультаты классификации:")
        print(f"Файл: {audio_path}")
        print(f"Предсказанный класс: {predicted_class}")
        print(f"Вероятность: {confidence:.2f}")

        # Выводим топ-3 предсказания
        print("\nТоп-3 предсказания:")
        top_indices = np.argsort(predictions[0])[::-1][:3]
        for i, idx in enumerate(top_indices):
            print(f"{i + 1}. {classes[idx]}: {predictions[0][idx]:.2f}")

    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")


def main_menu():
    while True:
        print("\n--- Меню классификации звуков ---")
        print("1. Создать новый датасет (запись с микрофона)")
        print("2. Создать и обучить новую нейросеть")
        print("3. Использовать нейросеть в реальном времени")
        print("4. Использовать нейросеть на заранее загруженном файле")
        print("5. Выйти")

        choice = input("Выберите опцию (1-5): ")

        if choice == '1':
            create_dataset()
        elif choice == '2':
            train_new_model()
        elif choice == '3':
            use_model_realtime()
        elif choice == '4':
            use_model_on_file()
        elif choice == '5':
            print("Выход из программы...")
            break
        else:
            print("Неверный ввод. Пожалуйста, выберите от 1 до 5.")


if __name__ == "__main__":
    print("Программа для классификации звуков с помощью нейросетей")
    main_menu()