from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path


class PoseTrainer:
    def __init__(self):
        self.pose_model = YOLO('yolo11s-pose.pt')
        self.classifier_model = None
        self.dataset_dir = Path("pose_dataset")
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


def main():
    trainer = PoseTrainer()

    while True:
        print("\n" + "=" * 50)
        print("1. Добавить новую позу в датасет")
        print("2. Запустить распознавание поз")
        print("3. Показать список сохраненных поз")
        print("4. Выйти")

        choice = input("Выберите действие: ")

        if choice == "1":
            trainer.capture_new_pose()
        elif choice == "2":
            trainer.run_detection()
        elif choice == "3":
            print("\nСохраненные позы:")
            for i, pose in enumerate(trainer.pose_data.keys(), 1):
                print(f"{i}. {pose}")
        elif choice == "4":
            print("Выход из программы...")
            break
        else:
            print("Неверный ввод, попробуйте еще раз")


if __name__ == "__main__":
    main()