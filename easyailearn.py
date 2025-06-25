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
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

#Class image
def process_classification_result(result):
    if hasattr(result, 'probs') and hasattr(result, 'names'):
        probs = result.probs.data.tolist()
        names = result.names
        return names, probs
    return None, None


def test_with_webcam():
    model = YOLO("best.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть веб камеру")
        return

    print("Нажмите 'q' для выхода из режима веб камеры")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с веб камеры")
            break

        results = model(frame)

        if results:
            result = results[0]
            names, probs = process_classification_result(result)

            if names and probs:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (300, 150), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

                top3_indices = np.argsort(probs)[-3:][::-1]

                y_offset = 30
                for i, idx in enumerate(top3_indices):
                    class_name = names[idx]
                    prob = probs[idx]
                    text = f"{i + 1}. {class_name}: {prob:.2f}"
                    color = (0, 255, 0) if i == 0 else (255, 255, 255)
                    cv2.putText(frame, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 30

                top1_idx = top3_indices[0]
                print(f"Распознано: {names[top1_idx]} ({probs[top1_idx]:.2f})", end='\r')

        cv2.imshow('Тест через веб камеру', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nРежим веб камеры завершен")

# Pose
class PoseTrainer:
    def __init__(self):
        self.pose_model = YOLO('yolo11s-pose.pt')
        self.classifier_model = None
        self.dataset_dir = Path("datasets/pose")
        self.pose_data = {}
        self.load_dataset()
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.current_pose = None
        self.distance_threshold = 0.5
        self.min_keypoints_match = 0.7

    def load_dataset(self):
        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir()
            print(f"Создана папка датасета: {self.dataset_dir}")
            return

        for pose_file in self.dataset_dir.glob("*.txt"):
            try:
                pose_name = pose_file.stem
                with open(pose_file, 'r', encoding='utf-8') as f:
                    points = []
                    for line in f:
                        coords = list(map(float, line.strip().split()))
                        points.append(coords)
                    self.pose_data[pose_name] = points
            except Exception as e:
                print(f"Ошибка загрузки {pose_file}: {e}")

        print(f"Загружено поз: {len(self.pose_data)}")

    def capture_new_pose(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть камеру")
            return

        print("Встаньте в позу и нажмите 's' для сохранения ('q' - отмена)")
        keypoints = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Ошибка чтения кадра")
                    break

                results = self.pose_model(frame, verbose=False)
                if results and len(results[0].keypoints) > 0:
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()
                    frame = results[0].plot()
                    cv2.putText(frame, "Нажмите 's' чтобы сохранить", (20, 40), self.font, 0.8, (0, 255, 0), 2)

                cv2.imshow("yolo_pose", frame)

                key = cv2.waitKey(1)
                if key == ord('s'):
                    if keypoints is not None:
                        pose_name = input("Введите название позы: ")
                        if pose_name:
                            self.save_pose(pose_name, keypoints)
                            break
                    else:
                        print("Поза не обнаружена!")
                elif key == ord('q') or cv2.getWindowProperty("yolo_pose", cv2.WND_PROP_VISIBLE) < 1:
                    print("Отмена сохранения")
                    break

        finally:
            if cv2.getWindowProperty("yolo_pose", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("yolo_pose")
            cap.release()
            cv2.waitKey(1)

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть камеру")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Ошибка чтения кадра")
                    break

                results = self.pose_model(frame, verbose=False)
                if results and len(results[0].keypoints) > 0:
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()
                    frame = results[0].plot()

                    pose, confidence = self.compare_poses(keypoints)

                    if pose and confidence > 0.5:
                        cv2.putText(frame, f"{pose} ({confidence:.2f})", (20, 50), self.font, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Скорректируйте позу", (20, 50), self.font, 0.8, (0, 0, 255), 2)

                cv2.imshow("yolo_pose", frame)

                if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty("yolo_pose", cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            if cv2.getWindowProperty("yolo_pose", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("yolo_pose")
            cap.release()
            cv2.waitKey(1)

    def save_pose(self, pose_name, keypoints):
        try:
            pose_file = self.dataset_dir / f"{pose_name}.txt"
            with open(pose_file, 'w', encoding='utf-8') as f:
                for kp in keypoints:
                    f.write(" ".join(map(str, kp)) + "\n")
            self.pose_data[pose_name] = keypoints
            print(f"Поза '{pose_name}' успешно сохранена!")
        except Exception as e:
            print(f"Ошибка при сохранении: {e}")

    def compare_poses(self, current_kp):
        current_norm = self.normalize_keypoints(current_kp)
        if current_norm is None:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for pose_name, ref_kp in self.pose_data.items():
            ref_kp_array = np.array(ref_kp)
            ref_norm = self.normalize_keypoints(ref_kp_array)
            if ref_norm is None:
                continue

            valid_points = 0
            total_distance = 0.0

            for i in range(min(len(current_norm), len(ref_norm))):
                if np.any(current_kp[i] == 0) or np.any(ref_kp_array[i] == 0):
                    continue

                distance = np.linalg.norm(current_norm[i] - ref_norm[i])
                total_distance += distance
                valid_points += 1

            if valid_points == 0:
                continue

            avg_distance = total_distance / valid_points
            score = 1.0 / (1.0 + avg_distance)

            if score > best_score and valid_points >= len(ref_norm) * self.min_keypoints_match:
                best_score = score
                best_match = pose_name

        return best_match, best_score

    def normalize_keypoints(self, keypoints):
        if len(keypoints) < 5:
            return None

        neck = (keypoints[5] + keypoints[6]) / 2
        normalized = keypoints - neck
        scale = np.linalg.norm(normalized[0]) + 1e-8
        return normalized / scale

# Audio
def create_dataset():
    dataset_path = f'datasets/{input("Введите имя для нового датасета (будет создана папка с этим именем): ")}'

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
    dataset_path = f'datasets/{input("Введите название датасета: ")}'
    if not os.path.exists(dataset_path):
        print("Ошибка: указанный путь не существует!")
        return
    epochs = int(input("Введите количество эпох обучения (рек. 100): "))
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

    model.save(f'runs/audio/{model_name}')

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


class TextModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, dataset_path, output_dir, epochs=3):
        try:
            os.makedirs(dataset_path, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            def load_text_data(path):
                files = [f for f in Path(path).glob("*.txt") if f.is_file()]
                texts = []
                for file in files:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                    except Exception as e:
                        print(f"Ошибка чтения файла {file}: {e}")
                return texts

            texts = load_text_data(dataset_path)
            if not texts:
                raise ValueError("Не найдены текстовые файлы в датасете")

            temp_file = os.path.join(output_dir, "temp_dataset.txt")
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(texts))

            model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

            class TextDataset(torch.utils.data.Dataset):
                def __init__(self, file_path, tokenizer):
                    self.tokenizer = tokenizer
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.texts = f.read().split('\n\n')

                def __len__(self):
                    return len(self.texts)

                def __getitem__(self, idx):
                    encoding = self.tokenizer(
                        self.texts[idx],
                        truncation=True,
                        max_length=128,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze()
                    }

            dataset = TextDataset(temp_file, self.tokenizer)

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=2,
                save_steps=500,
                save_total_limit=2,
                logging_dir=os.path.join(output_dir, 'logs'),
                fp16=torch.cuda.is_available(),
                learning_rate=3e-5,
                optim="adamw_torch",
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )

            trainer.train()
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            os.remove(temp_file)
            return True, "Модель успешно обучена!"

        except Exception as e:
            return False, f"Ошибка: {str(e)}"

    def load_model(self, model_dir):
        try:
            self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
            return True, "Модель загружена"
        except Exception as e:
            return False, f"Ошибка загрузки: {str(e)}"

    def generate_text(self, prompt, max_length=100):
        if not self.model or not self.tokenizer:
            return "Модель не загружена"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    while True:
        print("\n--- Меню Easy-AI-Learn ---")
        print("1. Классификация изображений")
        print("2. Распознование поз")
        print("3. Классификация аудио")
        print("4. Работа с текстом")
        print("5. Выйти")

        y = input("Выберите опцию (1-5): ")

        if y == '1':
            while True:
                print("\n--- Меню классификации изображений ---")
                print("1. Обучить новую модель")
                print("2. Тестировать на изображении")
                print("3. Тестировать через веб камеру")
                print("4. Назад")
                x = input("Выберите опцию (1-5): ")
                if x == '1':
                    model = YOLO("yolo11s-cls.pt")
                    z = input('Введите название dataset: ')
                    y = int(input('Количество эпох: '))
                    d = int(input('Устройство (0-GPU, 1-CPU(не рекомендуется)): '))
                    if d == 1:
                        d = 'cpu'

                    try:
                        results = model.train(data=z, epochs=y, imgsz=100, device=d)
                        print('Модель успешно обучена! Проверьте папку runs/classify/train/weights/best.pt')
                    except Exception as e:
                        print(f'Ошибка при обучении: {e}')

                elif x == '2':
                    z = input('Введите название изображения (с расширением): ')
                    model = YOLO("best.pt")
                    results = model(z)

                    if results:
                        result = results[0]
                        names, probs = process_classification_result(result)
                        if names and probs:
                            print("\nРезультаты классификации:")
                            top_idx = np.argmax(probs)
                            print(f"\nНаиболее вероятно: {names[top_idx]} ({probs[top_idx]:.2f})")

                elif x == '3':
                    test_with_webcam()

                elif x == '4':
                    break
                else:
                    print("Неверный ввод")

        elif y == '2':
            while True:
                trainer = PoseTrainer()
                print("\n--- Меню распознавания поз ---")
                print("1. Добавить новую позу в датасет")
                print("2. Запустить распознавание поз")
                print("3. Показать список сохраненных поз")
                print("4. Назад")

                choice = input("Выберите опцию (1-4): ")

                if choice == "1":
                    trainer.capture_new_pose()
                elif choice == "2":
                    trainer.run_detection()
                elif choice == "3":
                    print("\nСохраненные позы:")
                    for i, pose in enumerate(trainer.pose_data.keys(), 1):
                        print(f"{i}. {pose}")
                elif choice == "4":
                    break
                else:
                    print("Неверный ввод, попробуйте еще раз")

        elif y == '3':
            while True:
                print("\n--- Меню классификации звуков ---")
                print("1. Создать новый датасет (запись с микрофона)")
                print("2. Создать и обучить новую нейросеть")
                print("3. Использовать нейросеть в реальном времени")
                print("4. Использовать нейросеть на заранее загруженном файле")
                print("5. Назад")

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
                    break
                else:
                    print("Неверный ввод. Пожалуйста, выберите от 1 до 5.")

        elif y == '4':
            while True:
                text_model = TextModelHandler()
                print("\n--- Меню классификации звуков ---")
                print("1. Создать и обучить новую нейросеть")
                print("2. Использовать нейросеть")
                print("3. Назад")

                choice = input("Выберите опцию (1-3): ")

                if choice == '1':
                    dataset_path = "datasets/text"
                    output_dir = "runs/text"
                    x = int(input('Сколько эпох: '))
                    text_model.train(dataset_path, output_dir, x)
                elif choice == '2':
                    model_dir = "runs/text"
                    success, message = text_model.load_model(model_dir)
                    print(message)
                    if not text_model.model:
                        print("Сначала загрузите модель!")
                        continue
                    prompt = input("Введите начальный текст для генерации: ")
                    max_length = int(input("Введите максимальную длину текста (по умолчанию 100): ") or 100)
                    generated_text = text_model.generate_text(prompt, max_length)
                    print("\nСгенерированный текст:")
                    print(generated_text)
                elif choice == '3':
                    break
                else:
                    print("Неверный ввод. Пожалуйста, выберите от 1 до 5.")

        elif y == '5':
            print("Выход из программы")
            break
        else:
            print("Неверный ввод. Пожалуйста, выберите от 1 до 5.")


if __name__ == '__main__':
    main()