import cv2

# 1. Carica l'immagine
img_path = 'image.png'  # Sostituisci con il percorso della tua immagine
image = cv2.imread(img_path)

# 2. Controlla se l'immagine è stata caricata correttamente
if image is None:
    print("Errore nel caricamento dell'immagine.")
else:
    # 3. Visualizza l'immagine
    cv2.imshow('Immagine', image)

    # Aspetta che l'utente prema un tasto
    cv2.waitKey(0)  # 0 significa che aspetta indefinitamente

    # 4. Chiudi tutte le finestre
    cv2.destroyAllWindows()
