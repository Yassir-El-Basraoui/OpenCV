import cv2
import mediapipe as mp

# Inizializza MediaPipe per il rilevamento della mano
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Apri la fotocamera
cap = cv2.VideoCapture(0)

# Configura il rilevatore di mani
with mp_hands.Hands(max_num_hands=3, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Errore nel catturare il frame.")
            break

        # Specchia l'immagine per una visualizzazione più intuitiva
        frame = cv2.flip(frame, 1)

        # Converti l'immagine in RGB (richiesto da Mediapipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Esegui il rilevamento della mano
        results = hands.process(rgb_frame)

        # Se viene rilevata una mano
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Identifica se la mano è destra o sinistra
                hand_label = hand_info.classification[0].label
                is_right_hand = hand_label == "Right"

                # Disegna i punti e le linee della mano sull'immagine
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calcola il numero di dita alzate
                fingers_up = []

                # Elenco degli indici dei landmark per le punte delle dita
                tip_ids = [4, 8, 12, 16, 20]

                for tip_id in tip_ids:
                    if tip_id == 4:
                        # Logica per il pollice, diversa per mano destra o sinistra
                        if is_right_hand:
                            fingers_up.append(hand_landmarks.landmark[tip_id].x < hand_landmarks.landmark[tip_id - 1].x)
                        else:
                            fingers_up.append(hand_landmarks.landmark[tip_id].x > hand_landmarks.landmark[tip_id - 1].x)
                    else:
                        # Controllo per altre dita (indice, medio, anulare, mignolo)
                        fingers_up.append(hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y)

                # Conta il numero di dita alzate
                count_fingers = sum(fingers_up)

                # Visualizza il conteggio sullo schermo
                cv2.putText(frame, f"{hand_label} - Dita alzate: {count_fingers}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

        # Mostra il frame con il conteggio
        cv2.imshow("Hand Tracking - Conteggio delle Dita", frame)

        # Premi 'q' per uscire
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Rilascia la fotocamera e chiudi tutte le finestre
cap.release()
cv2.destroyAllWindows()
