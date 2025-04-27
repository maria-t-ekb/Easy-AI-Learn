from ultralytics import YOLO
import cv2
import numpy as np


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


def main():
    while True:
        print("\n" + "=" * 50)
        x = int(input(
            "Что будем делать:\n1. Обучить новую модель\n2. Тестировать на изображении\n3. Тестировать через веб камеру\n4. Выйти\nВыбор: "))

        if x == 1:
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

        elif x == 2:
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

        elif x == 3:
            test_with_webcam()

        elif x == 4:
            print("Выход из программы")
            break
        else:
            print("Неверный ввод")


if __name__ == '__main__':
    main()