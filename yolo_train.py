from easyailearn import *

def y_train(path, y=3, d=0):
    model = YOLO("yolo11s-cls.pt")
    z = path
    if d == 1:
        d = 'cpu'

    try:
        results = model.train(data=z, epochs=y, imgsz=100, device=d)

        return 'Модель успешно обучена!'    # Проверьте папку runs/classify/train/weights/best.pt'
    except Exception as e:
        return f'Ошибка при обучении: {e}'

def y_test(path):
    z = path
    model = YOLO("runs/classify/train/weights/best.pt")
    results = model(z)

    if results:
        result = results[0]
        names, probs = process_classification_result(result)
        if names and probs:
            # print("Результаты классификации:")
            top_idx = np.argmax(probs)
            # print(f"\nНаиболее вероятно: {names[top_idx]} ({probs[top_idx]:.2f})")
            return f'Наиболее вероятно: {names[top_idx]} ({probs[top_idx]:.2f})'

print('Hello')


def train_and_test(t, choice, path='', epoch=1, d=1):
    while True:
        # print("\n--- Меню Easy-AI-Learn ---")
        # print("1. Классификация изображений")
        # print("2. Распознование поз")
        # print("3. Классификация аудио")
        # print("4. Работа с текстом")
        # print("5. Выйти")
        if t == 'image':
            # print("\n--- Меню классификации изображений ---")
            # print("1. Обучить новую модель")
            # print("2. Тестировать на изображении")
            # print("3. Тестировать через веб камеру")
            # print("4. Назад")
            if choice == '1':
                model = YOLO("yolo11s-cls.pt")
                # z = input('Введите название dataset: ')
                # y = int(input('Количество эпох: '))
                # d = int(input('Устройство (0-GPU, 1-CPU(не рекомендуется)): '))
                if d == 1:
                    d = 'cpu'

                try:
                    results = model.train(data=path, epochs=epoch, imgsz=100, device=d)
                    print('Модель успешно обучена! Проверьте папку runs/classify/train/weights/best.pt')
                except Exception as e:
                    print(f'Ошибка при обучении: {e}')

            elif choice == '2':
                # z = input('Введите название изображения (с расширением): ')
                model = YOLO("best.pt")
                results = model(path)

                if results:
                    result = results[0]
                    names, probs = process_classification_result(result)
                    if names and probs:
                        print("\nРезультаты классификации:")
                        top_idx = np.argmax(probs)
                        print(f"\nНаиболее вероятно: {names[top_idx]} ({probs[top_idx]:.2f})")

        elif t == 'pose':
            trainer = PoseTrainer()
            # print("\n--- Меню распознавания поз ---")
            # print("1. Добавить новую позу в датасет")
            # print("2. Запустить распознавание поз")
            # print("3. Показать список сохраненных поз")
            # print("4. Назад")

            # choice = input("Выберите опцию (1-4): ")

            if choice == "1":
                trainer.capture_new_pose()
            elif choice == "2":
                trainer.run_detection()
            # elif choice == "3":
            #     print("\nСохраненные позы:")
            #     for i, pose in enumerate(trainer.pose_data.keys(), 1):
            #         print(f"{i}. {pose}")

        elif t == 'audio':
                # print("\n--- Меню классификации звуков ---")
                # print("1. Создать новый датасет (запись с микрофона)")
                # print("2. Создать и обучить новую нейросеть")
                # print("3. Использовать нейросеть в реальном времени")
                # print("4. Использовать нейросеть на заранее загруженном файле")
                # print("5. Назад")

                # choice = input("Выберите опцию (1-5): ")

                # if choice == '1':
                #     create_dataset()
                if choice == '1':
                    train_new_model()
                # elif choice == '3':
                #     use_model_realtime()
                elif choice == '2':
                    use_model_on_file()

        elif t == 'text':
            text_model = TextModelHandler()
            # print("\n--- Меню классификации звуков ---")
            # print("1. Создать и обучить новую нейросеть")
            # print("2. Использовать нейросеть")
            # print("3. Назад")

            # choice = input("Выберите опцию (1-3): ")

            if choice == '1':
                dataset_path = "datasets/text"
                output_dir = "runs/text"
                # x = int(input('Сколько эпох: '))
                text_model.train(dataset_path, output_dir, epoch)
            elif choice == '2':
                try:
                    model_dir = "runs/text"
                    success, message = text_model.load_model(model_dir)
                    # return message
                    if not text_model.model:
                        # print("Сначала загрузите модель!")
                        continue
                    prompt = input("Введите начальный текст для генерации: ")
                    max_length = int(input("Введите максимальную длину текста (по умолчанию 100): ") or 100)
                    generated_text = text_model.generate_text(prompt, max_length)
                    # print("\nСгенерированный текст:")
                    return generated_text
                except:
                    return ' '
