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

print('Hello')


def main(y, ):
        if y == '1':
            # print("\n--- Меню классификации изображений ---")
            # print("1. Обучить новую модель")
            # print("2. Тестировать на изображении")
            # print("3. Тестировать через веб камеру")
            # print("4. Назад")
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
                    return 'Модель успешно обучена! Проверьте папку runs/classify/train/weights/best.pt'
                except Exception as e:
                    return f'Ошибка при обучении: {e}'

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
                        return f"\nНаиболее вероятно: {names[top_idx]} ({probs[top_idx]:.2f})"
        elif y == '2':
                trainer = PoseTrainer()
                # print("\n--- Меню распознавания поз ---")
                # print("1. Добавить новую позу в датасет")
                # print("2. Запустить распознавание поз")
                # print("3. Показать список сохраненных поз")
                # print("4. Назад")

                choice = input("Выберите опцию (1-4): ")

                if choice == "1":
                    trainer.capture_new_pose()
                elif choice == "2":
                    trainer.run_detection()

        elif y == '3':
                # print("\n--- Меню классификации звуков ---")
                # print("1. Создать новый датасет (запись с микрофона)")
                # print("2. Создать и обучить новую нейросеть")
                # print("3. Использовать нейросеть в реальном времени")
                # print("4. Использовать нейросеть на заранее загруженном файле")
                # print("5. Назад")

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
            print('В разработке')

