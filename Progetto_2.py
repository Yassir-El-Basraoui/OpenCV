import cv2

# 1. Carica l'immagine
img_path = 'image.png'  # Sostituisci con il percorso della tua immagine
image = cv2.imread(img_path)

# 2. Controlla se l'immagine è stata caricata correttamente
if image is None:
    print("Errore nel caricamento dell'immagine.")
else:
    # 3. Converti l'immagine a scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Applica un filtro di sfocatura
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 5. Rileva i bordi utilizzando Canny
    edges = cv2.Canny(blurred_image, 100, 200)

    # 6. Visualizza le immagini
    cv2.imshow('Immagine Originale', image)
    cv2.imshow('Immagine Grigia', gray_image)
    cv2.imshow('Immagine Sfocata', blurred_image)
    cv2.imshow('Bordi Rilevati', edges)

    # Aspetta che l'utente prema un tasto
    cv2.waitKey(0)

    # 7. Chiudi tutte le finestre
    cv2.destroyAllWindows()
