from tkinter import ttk, messagebox, filedialog, scrolledtext
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
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
from ultralytics import YOLO
import cv2
from pathlib import Path
from PIL import Image, ImageTk


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

class LoadingScreen:
    def __init__(self, root, title="Загрузка..."):
        self.top = tk.Toplevel(root)
        self.top.title(title)
        self.top.geometry("300x100")
        self.top.resizable(False, False)
        root_x = root.winfo_x()
        root_y = root.winfo_y()
        root_width = root.winfo_width()
        root_height = root.winfo_height()
        x = root_x + (root_width - 300) // 2
        y = root_y + (root_height - 100) // 2
        self.top.geometry(f"+{x}+{y}")
        self.label = tk.Label(self.top, text=title, font=('Arial', 12))
        self.label.pack(pady=10)
        self.progress = ttk.Progressbar(self.top, mode='indeterminate')
        self.progress.pack(pady=5)
        self.progress.start()
        self.top.grab_set()

    def close(self):
        self.progress.stop()
        self.top.destroy()


class EasyAILearnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Easy AI Learn")
        self.root.geometry("800x600")
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10), padding=5)
        self.create_main_menu()
        self.pose_trainer = None
        self.audio_model = None
        self.audio_classes = None
        self.image_model = None
        self.image_label = None
        self.canvas = None
        self.photo = None
        self.text_model_handler = TextModelHandler()

    def create_main_menu(self):
        self.clear_frame()
        title_label = tk.Label(self.root, text="Easy AI Learn", font=('Arial', 20, 'bold'))
        title_label.pack(pady=20)
        buttons = [
            ("Классификация изображений", self.show_image_menu),
            ("Распознавание поз", self.show_pose_menu),
            ("Классификация аудио", self.show_audio_menu),
            ("Работа с текстом", self.show_text_menu),
            ("Выход", self.root.quit)
        ]

        for text, command in buttons:
            btn = ttk.Button(self.root, text=text, command=command, width=30)
            btn.pack(pady=5)

    def show_text_menu(self):
        self.clear_frame()
        title_label = tk.Label(self.root, text="Работа с текстом", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        buttons = [
            ("Обучить модель на текстах", self.train_text_model),
            ("Протестировать модель", self.test_text_model),
            ("Назад", self.back_to_main)
        ]

        for text, command in buttons:
            btn = ttk.Button(self.root, text=text, command=command, width=30)
            btn.pack(pady=5)

    def train_text_model(self):
        self.clear_frame()
        title_label = tk.Label(self.root, text="Обучение текстовой модели", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Label(frame, text="Количество эпох:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.text_epochs_entry = tk.Entry(frame, width=30)
        self.text_epochs_entry.insert(0, "3")
        self.text_epochs_entry.grid(row=0, column=1, padx=5, pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Начать обучение", command=self.start_text_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Назад", command=self.show_text_menu).pack(side=tk.LEFT, padx=5)

    def start_text_training(self):
        epochs = self.text_epochs_entry.get()

        try:
            epochs = int(epochs)
        except ValueError:
            messagebox.showerror("Ошибка", "Количество эпох должно быть числом")
            return

        loading = LoadingScreen(self.root, "Обучение текстовой модели...")

        def train_thread():
            dataset_path = "datasets/text"
            output_dir = "runs/text"

            success, message = self.text_model_handler.train(dataset_path, output_dir, epochs)

            self.root.after(0, loading.close)
            if success:
                self.root.after(0, lambda: messagebox.showinfo("Успех", message))
            else:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", message))
            self.root.after(0, self.show_text_menu)

        threading.Thread(target=train_thread, daemon=True).start()

    def test_text_model(self):
        self.clear_frame()
        title_label = tk.Label(self.root, text="Тестирование текстовой модели", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Загрузка модели
        model_dir = "runs/text"
        if not os.path.exists(model_dir):
            messagebox.showerror("Ошибка", "Модель не найдена. Сначала обучите модель.")
            self.show_text_menu()
            return

        success, message = self.text_model_handler.load_model(model_dir)
        if not success:
            messagebox.showerror("Ошибка", message)
            self.show_text_menu()
            return

        tk.Label(self.root, text="Введите промпт для генерации текста:").pack(pady=5)

        self.text_prompt_entry = tk.Entry(self.root, width=60)
        self.text_prompt_entry.pack(pady=5)

        generate_btn = ttk.Button(self.root, text="Сгенерировать текст", command=self.generate_text_response)
        generate_btn.pack(pady=10)

        self.text_output = scrolledtext.ScrolledText(self.root, width=80, height=15, wrap=tk.WORD)
        self.text_output.pack(pady=10)

        ttk.Button(self.root, text="Назад", command=self.show_text_menu).pack(pady=10)

    def generate_text_response(self):
        prompt = self.text_prompt_entry.get()
        if not prompt:
            messagebox.showwarning("Предупреждение", "Введите текст для генерации")
            return

        loading = LoadingScreen(self.root, "Генерация текста...")

        def generate_thread():
            try:
                response = self.text_model_handler.generate_text(prompt)
                self.root.after(0, lambda: self.display_generated_text(response))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка генерации: {str(e)}"))
            finally:
                self.root.after(0, loading.close)

        threading.Thread(target=generate_thread, daemon=True).start()

    def display_generated_text(self, text):
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.INSERT, text)

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def back_to_main(self):
        self.create_main_menu()

    def show_image_menu(self):
        self.clear_frame()
        title_label = tk.Label(self.root, text="Классификация изображений", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        buttons = [
            ("Обучить новую модель", self.train_image_model),
            ("Тестировать на изображении", self.test_image_file),
            ("Тестировать через веб-камеру", self.test_image_webcam),
            ("Назад", self.back_to_main)
        ]

        for text, command in buttons:
            btn = ttk.Button(self.root, text=text, command=command, width=30)
            btn.pack(pady=5)

    def show_pose_menu(self):
        self.clear_frame()

        title_label = tk.Label(self.root, text="Распознавание поз", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        buttons = [
            ("Добавить новую позу", self.add_new_pose),
            ("Запустить распознавание поз", self.run_pose_detection),
            ("Список сохраненных поз", self.show_saved_poses),
            ("Назад", self.back_to_main)
        ]

        for text, command in buttons:
            btn = ttk.Button(self.root, text=text, command=command, width=30)
            btn.pack(pady=5)

    def show_audio_menu(self):
        self.clear_frame()
        title_label = tk.Label(self.root, text="Классификация аудио", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        buttons = [
            ("Создать новый датасет", self.create_audio_dataset),
            ("Обучить новую модель", self.train_audio_model),
            ("Тестировать в реальном времени", self.test_audio_realtime),
            ("Тестировать на файле", self.test_audio_file),
            ("Назад", self.back_to_main)
        ]

        for text, command in buttons:
            btn = ttk.Button(self.root, text=text, command=command, width=30)
            btn.pack(pady=5)

    def train_image_model(self):
        self.clear_frame()

        title_label = tk.Label(self.root, text="Обучение модели классификации изображений", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        frame = tk.Frame(self.root)
        frame.pack(pady=10)
        tk.Label(frame, text="Название датасета:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.dataset_entry = tk.Entry(frame, width=30)
        self.dataset_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(frame, text="Количество эпох:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.epochs_entry = tk.Entry(frame, width=30)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=1, column=1, padx=5, pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Начать обучение", command=self.start_image_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Назад", command=self.show_image_menu).pack(side=tk.LEFT, padx=5)

    def start_image_training(self):
        dataset = self.dataset_entry.get()
        epochs = self.epochs_entry.get()
        device = 'cpu'

        if not dataset or not epochs:
            messagebox.showerror("Ошибка", "Пожалуйста, заполните все поля")
            return

        try:
            epochs = int(epochs)
        except ValueError:
            messagebox.showerror("Ошибка", "Количество эпох должно быть числом")
            return

        loading = LoadingScreen(self.root, "Обучение модели...")

        def train_thread():
            try:
                model = YOLO("yolo11s-cls.pt")
                results = model.train(data=dataset, epochs=epochs, imgsz=100, device='cpu')

                self.root.after(0, loading.close)
                self.root.after(0, lambda: messagebox.showinfo("Успех",
                                                               "Модель успешно обучена! Проверьте папку runs/classify/train/weights/best.pt"))
                self.root.after(0, self.show_image_menu)
            except Exception as e:
                self.root.after(0, loading.close)
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка при обучении: {str(e)}"))

        threading.Thread(target=train_thread, daemon=True).start()

    def test_image_file(self):
        filepath = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not filepath:
            return

        loading = LoadingScreen(self.root, "Обработка изображения...")

        def process_thread():
            try:
                model = YOLO("best.pt")
                results = model(filepath)

                if results:
                    result = results[0]
                    names = result.names
                    probs = result.probs.data.tolist()

                    if names and probs:
                        top_idx = np.argmax(probs)
                        img = cv2.imread(filepath)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img)
                        img.thumbnail((600, 600))

                        self.root.after(0, loading.close)
                        self.show_image_results(img, names, probs)

            except Exception as e:
                self.root.after(0, loading.close)
                self.root.after(0,
                                lambda: messagebox.showerror("Ошибка", f"Ошибка при обработке изображения: {str(e)}"))

        threading.Thread(target=process_thread, daemon=True).start()

    def show_image_results(self, img, names, probs):
        self.clear_frame()
        title_label = tk.Label(self.root, text="Результаты классификации", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)

        if not hasattr(self, 'canvas'):
            self.canvas = tk.Canvas(self.root, width=600, height=400)
            self.canvas.pack(pady=10)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10)
        top_idx = np.argmax(probs)
        tk.Label(result_frame, text=f"Наиболее вероятно: {names[top_idx]} ({probs[top_idx]:.2f})",
                 font=('Arial', 12, 'bold')).pack()
        tk.Label(result_frame, text="\nТоп-3 предсказания:", font=('Arial', 11)).pack()
        top_indices = np.argsort(probs)[-3:][::-1]

        for i, idx in enumerate(top_indices):
            tk.Label(result_frame, text=f"{i + 1}. {names[idx]}: {probs[idx]:.2f}").pack()

        ttk.Button(self.root, text="Назад", command=self.show_image_menu).pack(pady=10)

    def test_image_webcam(self):
        messagebox.showinfo("Информация", "Для выхода из режима веб-камеры нажмите 'q'")

        def webcam_thread():
            model = YOLO("best.pt")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось открыть веб-камеру"))
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)

                if results:
                    result = results[0]
                    names, probs = result.names, result.probs.data.tolist()

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

                cv2.imshow('Тест через веб камеру', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=webcam_thread, daemon=True).start()

    def add_new_pose(self):
        if not self.pose_trainer:
            self.pose_trainer = PoseTrainer()

        pose_name = tk.simpledialog.askstring("Новая поза", "Введите название позы:")
        if not pose_name:
            return

        messagebox.showinfo("Информация", "Встаньте в позу и нажмите 's' для сохранения ('q' - отмена)")

        def capture_thread():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось открыть камеру"))
                return

            keypoints = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.pose_trainer.pose_model(frame, verbose=False)
                if results and len(results[0].keypoints) > 0:
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()
                    frame = results[0].plot()
                    cv2.putText(frame, "Нажмите 's' чтобы сохранить", (20, 40),
                                self.pose_trainer.font, 0.8, (0, 255, 0), 2)

                cv2.imshow("yolo_pose", frame)

                key = cv2.waitKey(1)
                if key == ord('s'):
                    if keypoints is not None:
                        self.pose_trainer.save_pose(pose_name, keypoints)
                        self.root.after(0, lambda: messagebox.showinfo("Успех", f"Поза '{pose_name}' сохранена!"))
                        break
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Ошибка", "Поза не обнаружена!"))
                elif key == ord('q') or cv2.getWindowProperty("yolo_pose", cv2.WND_PROP_VISIBLE) < 1:
                    self.root.after(0, lambda: messagebox.showinfo("Информация", "Отмена сохранения"))
                    break

            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        threading.Thread(target=capture_thread, daemon=True).start()

    def run_pose_detection(self):
        if not self.pose_trainer:
            self.pose_trainer = PoseTrainer()

        messagebox.showinfo("Информация", "Для выхода нажмите 'q'")

        def detection_thread():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось открыть камеру"))
                return

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = self.pose_trainer.pose_model(frame, verbose=False)
                    if results and len(results[0].keypoints) > 0:
                        keypoints = results[0].keypoints.xy[0].cpu().numpy()
                        frame = results[0].plot()

                        pose, confidence = self.pose_trainer.compare_poses(keypoints)

                        if pose and confidence > 0.5:
                            cv2.putText(frame, f"{pose} ({confidence:.2f})", (20, 50),
                                        self.pose_trainer.font, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Скорректируйте позу", (20, 50),
                                        self.pose_trainer.font, 0.8, (0, 0, 255), 2)

                    cv2.imshow("yolo_pose", frame)

                    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty("yolo_pose", cv2.WND_PROP_VISIBLE) < 1:
                        break

            finally:
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)

        threading.Thread(target=detection_thread, daemon=True).start()

    def show_saved_poses(self):
        if not self.pose_trainer:
            self.pose_trainer = PoseTrainer()

        if not self.pose_trainer.pose_data:
            messagebox.showinfo("Информация", "Нет сохраненных поз")
            return

        self.clear_frame()
        title_label = tk.Label(self.root, text="Сохраненные позы", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        listbox = tk.Listbox(self.root, width=50, height=15, font=('Arial', 12))
        listbox.pack(pady=10)

        for pose in self.pose_trainer.pose_data.keys():
            listbox.insert(tk.END, pose)

        ttk.Button(self.root, text="Назад", command=self.show_pose_menu).pack(pady=10)

    def create_audio_dataset(self):
        dataset_name = tk.simpledialog.askstring("Новый датасет", "Введите имя для нового датасета:")
        if not dataset_name:
            return

        dataset_path = f'datasets/{dataset_name}'

        if os.path.exists(dataset_path):
            messagebox.showerror("Ошибка", "Папка уже существует!")
            return

        os.makedirs(dataset_path)

        messagebox.showinfo("Информация", "Создание нового датасета. Введите 'стоп' для завершения.")

        def recording_thread():
            try:
                while True:
                    class_name = tk.simpledialog.askstring("Новый класс",
                                                           "Введите имя нового класса (или 'стоп' для завершения):")
                    if not class_name or class_name.lower() == 'стоп':
                        break

                    class_path = os.path.join(dataset_path, class_name)
                    os.makedirs(class_path, exist_ok=True)

                    sample_count = 0
                    while True:
                        confirm = messagebox.askyesno("Запись",
                                                      f"Класс '{class_name}'. Нажмите OK для записи образца {sample_count + 1} или Отмена для перехода к следующему классу.")
                        if not confirm:
                            break

                        try:
                            duration = 3
                            sample_rate = 22050
                            messagebox.showinfo("Запись", f"Записываю {duration} секунды... (говорите сейчас)")

                            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
                            sd.wait()
                            audio = np.squeeze(audio)
                            filename = os.path.join(class_path, f"sample_{sample_count}.wav")
                            sf.write(filename, audio, sample_rate)

                            messagebox.showinfo("Успех", f"Образец сохранен как {filename}")
                            sample_count += 1
                        except Exception as e:
                            messagebox.showerror("Ошибка", f"Ошибка при записи: {e}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при создании датасета: {e}")

        threading.Thread(target=recording_thread, daemon=True).start()

    def train_audio_model(self):
        self.clear_frame()

        title_label = tk.Label(self.root, text="Обучение модели классификации аудио", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        frame = tk.Frame(self.root)
        frame.pack(pady=10)
        tk.Label(frame, text="Название датасета:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.audio_dataset_entry = tk.Entry(frame, width=30)
        self.audio_dataset_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(frame, text="Количество эпох:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.audio_epochs_entry = tk.Entry(frame, width=30)
        self.audio_epochs_entry.insert(0, "100")
        self.audio_epochs_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Label(frame, text="Имя модели:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.audio_model_entry = tk.Entry(frame, width=30)
        self.audio_model_entry.grid(row=2, column=1, padx=5, pady=5)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Начать обучение", command=self.start_audio_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Назад", command=self.show_audio_menu).pack(side=tk.LEFT, padx=5)

    def start_audio_training(self):
        dataset = self.audio_dataset_entry.get()
        epochs = self.audio_epochs_entry.get()
        model_name = self.audio_model_entry.get()

        if not dataset or not epochs or not model_name:
            messagebox.showerror("Ошибка", "Пожалуйста, заполните все поля")
            return

        try:
            epochs = int(epochs)
        except ValueError:
            messagebox.showerror("Ошибка", "Количество эпох должно быть числом")
            return

        dataset_path = f'datasets/{dataset}'
        if not os.path.exists(dataset_path):
            messagebox.showerror("Ошибка", "Указанный путь не существует!")
            return

        model_name = model_name + ".h5"

        loading = LoadingScreen(self.root, "Обучение модели...")

        def train_thread():
            try:
                print("Загрузка датасета...")
                features, labels = load_dataset(dataset_path)
                le = LabelEncoder()
                labels_encoded = le.fit_transform(labels)
                labels_categorical = to_categorical(labels_encoded)

                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels_categorical, test_size=0.2, random_state=42
                )

                input_shape = (X_train.shape[1],)
                num_classes = len(le.classes_)
                model = create_model(input_shape, num_classes)
                print("Обучение модели...")
                model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
                os.makedirs("runs/audio", exist_ok=True)
                model.save(f'runs/audio/{model_name}')
                np.save(f'runs/audio/{model_name.replace(".h5", "_classes.npy")}', le.classes_)

                self.root.after(0, loading.close)
                self.root.after(0, lambda: messagebox.showinfo(
                    "Успех",
                    f"Модель сохранена как {model_name}\nКлассы: {', '.join(le.classes_)}"
                ))
                self.root.after(0, self.show_audio_menu)

            except Exception as e:
                self.root.after(0, loading.close)
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка при обучении: {str(e)}"))

        threading.Thread(target=train_thread, daemon=True).start()

    def test_audio_realtime(self):
        model_path = filedialog.askopenfilename(
            title="Выберите файл модели",
            initialdir="runs/audio",
            filetypes=[("H5 files", "*.h5")]
        )

        if not model_path:
            return

        classes_path = model_path.replace('.h5', '_classes.npy')
        if not os.path.exists(classes_path):
            messagebox.showerror("Ошибка", "Файл классов не найден!")
            return

        try:
            self.audio_model = tf.keras.models.load_model(model_path)
            self.audio_classes = np.load(classes_path)

            messagebox.showinfo("Информация",
                                "Модель загружена. Нажмите OK для начала записи (Ctrl+C в консоли для выхода).")

            def recording_thread():
                try:
                    while True:
                        audio, sample_rate = record_audio()
                        features = extract_features(audio, sample_rate)
                        features = features.reshape(1, -1)

                        predictions = self.audio_model.predict(features, verbose=0)
                        predicted_index = np.argmax(predictions)
                        predicted_class = self.audio_classes[predicted_index]
                        confidence = predictions[0][predicted_index]

                        self.root.after(0, lambda: messagebox.showinfo(
                            "Результат",
                            f"Предсказанный класс: {predicted_class}\nВероятность: {confidence:.2f}"
                        ))

                except KeyboardInterrupt:
                    pass
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка при записи: {str(e)}"))

            threading.Thread(target=recording_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке модели: {str(e)}")

    def test_audio_file(self):
        model_path = filedialog.askopenfilename(
            title="Выберите файл модели",
            initialdir="runs/audio",
            filetypes=[("H5 files", "*.h5")]
        )

        if not model_path:
            return

        audio_path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Audio files", "*.wav *.mp3 *.ogg")]
        )

        if not audio_path:
            return

        classes_path = model_path.replace('.h5', '_classes.npy')
        if not os.path.exists(classes_path):
            messagebox.showerror("Ошибка", "Файл классов не найден!")
            return

        loading = LoadingScreen(self.root, "Обработка аудио...")

        def process_thread():
            try:
                model = tf.keras.models.load_model(model_path)
                classes = np.load(classes_path)
                data, sample_rate = librosa.load(audio_path)
                features = extract_features(data, sample_rate)
                features = features.reshape(1, -1)
                predictions = model.predict(features, verbose=0)
                predicted_index = np.argmax(predictions)
                predicted_class = classes[predicted_index]
                confidence = predictions[0][predicted_index]
                top_indices = np.argsort(predictions[0])[::-1][:3]
                top_results = []
                for i, idx in enumerate(top_indices):
                    top_results.append(f"{i + 1}. {classes[idx]}: {predictions[0][idx]:.2f}")

                self.root.after(0, loading.close)
                self.root.after(0, lambda: self.show_audio_results(
                    audio_path, predicted_class, confidence, top_results
                ))

            except Exception as e:
                self.root.after(0, loading.close)
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка при обработке файла: {str(e)}"))

        threading.Thread(target=process_thread, daemon=True).start()

    def show_audio_results(self, audio_path, predicted_class, confidence, top_results):
        self.clear_frame()

        title_label = tk.Label(self.root, text="Результаты классификации аудио", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)

        tk.Label(self.root, text=f"Файл: {os.path.basename(audio_path)}", font=('Arial', 12)).pack(pady=5)

        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10)

        tk.Label(result_frame, text="Предсказанный класс:", font=('Arial', 12)).grid(row=0, column=0, sticky='e')
        tk.Label(result_frame, text=predicted_class, font=('Arial', 12, 'bold')).grid(row=0, column=1, sticky='w')

        tk.Label(result_frame, text="Вероятность:", font=('Arial', 12)).grid(row=1, column=0, sticky='e')
        tk.Label(result_frame, text=f"{confidence:.2f}", font=('Arial', 12, 'bold')).grid(row=1, column=1, sticky='w')

        tk.Label(self.root, text="\nТоп-3 предсказания:", font=('Arial', 12)).pack()

        for result in top_results:
            tk.Label(self.root, text=result).pack()

        ttk.Button(self.root, text="Назад", command=self.show_audio_menu).pack(pady=20)


class PoseTrainer:
    def __init__(self):
        self.pose_model = YOLO('yolo11s-pose.pt').to('cpu')  # Загружаем модель на CPU
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


def main():
    root = tk.Tk()
    app = EasyAILearnApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
