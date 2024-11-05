import cv2

# 1. Apri la fotocamera
cap = cv2.VideoCapture(0)  # Usa l'indice 0 per la fotocamera frontale principale

# 2. Controlla se la fotocamera è stata aperta correttamente
if not cap.isOpened():
    print("Errore nell'apertura della fotocamera.")
else:
    print("Premi 'q' per chiudere il flusso video.")

# 3. Cattura i frame in un ciclo
while cap.isOpened():
    # Cattura ogni frame
    ret, frame = cap.read()
    
    if not ret:
        print("Errore nel catturare il frame.")
        break

    # 4. Converti il frame a scala di grigi
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Applica il rilevamento dei bordi di Canny
    edges = cv2.Canny(gray_frame, threshold1=50, threshold2=150)

    # 6. Mostra il frame originale e il frame con i bordi rilevati
    cv2.imshow('Frame Originale', frame)
    cv2.imshow('Bordi Rilevati (Canny)', edges)

    # 7. Esci dal ciclo premendo 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Rilascia la fotocamera e chiudi le finestre
cap.release()
cv2.destroyAllWindows()
