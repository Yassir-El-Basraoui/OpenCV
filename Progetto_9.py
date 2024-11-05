import cv2
import mediapipe as mp

# Inizializza MediaPipe per il rilevamento del volto
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Apri la fotocamera
cap = cv2.VideoCapture(0)

# Variabili per il conteggio di quando chiudi gli occhi
blink_count = 0
eye_closed = False

# Configura Face Mesh
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Errore nel catturare il frame.")
            break

        # Specchia l'immagine per una visualizzazione più intuitiva
        frame = cv2.flip(frame, 1)

        # Converti in RGB per MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Esegui il rilevamento del volto
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Disegna i landmark del viso
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )

                # Indici dei landmark degli occhi per controllare lo stato aperto/chiuso
                left_eye_top = face_landmarks.landmark[159]
                left_eye_bottom = face_landmarks.landmark[145]
                right_eye_top = face_landmarks.landmark[386]
                right_eye_bottom = face_landmarks.landmark[374]

                # Calcola la distanza tra la palpebra superiore e inferiore per entrambi gli occhi
                left_eye_distance = abs(left_eye_top.y - left_eye_bottom.y)
                right_eye_distance = abs(right_eye_top.y - right_eye_bottom.y)

                # Soglia per considerare l'occhio chiuso
                eye_closed_threshold = 0.02

                # Verifica se entrambi gli occhi sono chiusi
                if left_eye_distance < eye_closed_threshold and right_eye_distance < eye_closed_threshold:
                    if not eye_closed:
                        blink_count += 1
                        eye_closed = True
                else:
                    eye_closed = False

                # Mostra il conteggio dei blink sullo schermo
                cv2.putText(frame, f"Blink Count: {blink_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Mostra il frame con il conteggio dei blink
        cv2.imshow("Face Blink Detection", frame)

        # Premi 'q' per uscire
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Rilascia la fotocamera e chiudi tutte le finestre
cap.release()
cv2.destroyAllWindows()
