import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json

CATEGORIES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

class HandProcessor:
    def __init__(self, model_architecture="model/model_architecture.json", model_weights="model/ASL_model.weights.h5"):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.model = self.load_model(model_architecture, model_weights)

    @staticmethod
    def load_model(model_architecture, model_weights):
        with open(model_architecture, "r") as json_file:
            model = json_file.read()
        model = model_from_json(model)
        model.load_weights(model_weights)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def process_webcam_input(self, webcam_index=0):
        cap = cv2.VideoCapture(webcam_index)
        with self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                cropped_hands, all_landmarks, hand_landmarks, expanded_bbox = self.get_cropped_hands(image, hands)

                if hand_landmarks:
                    for hand_landmark in hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmark,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                    if hand_landmarks[0].landmark[5].x < hand_landmarks[0].landmark[17].x:
                        isLeft = True
                    else:
                        isLeft = False

                if len(cropped_hands) == 1:
                    self.display_results(image, cropped_hands, all_landmarks, expanded_bbox, isLeft)

                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_cropped_hands(image, hands):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        cropped_hands = []
        all_landmarks = []
        rescaled_landmarks = []
        expanded_bbox = ()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x * image.shape[1], lm.y * image.shape[0], lm.z) for lm in hand_landmarks.landmark]
                all_landmarks.extend(landmarks)

            if len(all_landmarks) > 0:
                all_landmarks = np.array(all_landmarks)
                min_x, min_y, _ = np.min(all_landmarks, axis=0)
                max_x, max_y, _ = np.max(all_landmarks, axis=0)
                expanded_bbox = (int(min_x) - 60, int(min_y) - 60, int(max_x) + 60, int(max_y) + 60)
                cropped_image = image[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cropped_image = cv2.resize(cropped_image, (128, 128))
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    cropped_hands.append(cropped_image)

            for landmark in landmarks:
                rescaled_landmark_x = int(
                    (landmark[0] - expanded_bbox[0]) * 128 / (expanded_bbox[2] - expanded_bbox[0]))
                rescaled_landmark_y = int(
                    (landmark[1] - expanded_bbox[1]) * 128 / (expanded_bbox[3] - expanded_bbox[1]))
                rescaled_landmarks.append((rescaled_landmark_x, rescaled_landmark_y, landmark[2]))

        return cropped_hands, rescaled_landmarks, results.multi_hand_landmarks, expanded_bbox

    def display_results(self, image, cropped_hands, landmarks, expanded_bbox, isLeft = False):
        label = ''

        if not isLeft:
            image_width = cropped_hands[0].shape[1]
            landmarks = [(image_width - x, y, z) for x, y, z in landmarks]
            cropped_hands[0] = cv2.flip(cropped_hands[0], 1)

        blur = cv2.GaussianBlur(cropped_hands[0], (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, threshold_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        mBlur_image = cv2.medianBlur(threshold_image, 5)

        rescaled_landmark_image = np.zeros((128, 128), dtype=np.uint8)
        for landmark in landmarks:
            cv2.circle(rescaled_landmark_image, (landmark[0], landmark[1]), 5, 255, -1)

        prediction = self.model.predict(
            [np.expand_dims(mBlur_image, axis=0), np.expand_dims(np.array(landmarks).reshape((21, 3, 1)), axis=0)])
        label = CATEGORIES[np.argmax(prediction)]

        display_image = np.hstack((cropped_hands[0], rescaled_landmark_image, th3, threshold_image, mBlur_image))

        cv2.rectangle(image, (expanded_bbox[0], expanded_bbox[1]), (expanded_bbox[2], expanded_bbox[3]), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        hand = 'Left' if isLeft else 'Right'
        text = f'hand: {hand}, label: {label}'
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = expanded_bbox[0] + (expanded_bbox[2] - expanded_bbox[0]) // 2 - text_size[0] // 2
        text_y = expanded_bbox[1] - 10
        cv2.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        overlay = image.copy()
        overlay[:128, :128 * 5, :] = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        alpha = 1
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


if __name__ == "__main__":
    hand_processor = HandProcessor()
    hand_processor.process_webcam_input(2)
