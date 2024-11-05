import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Carica l'immagine
img_path = 'image.png'  # Sostituisci con il percorso della tua immagine
image = cv2.imread(img_path)

# 2. Verifica che l'immagine sia stata caricata correttamente
if image is None:
    print("Errore nel caricamento dell'immagine.")
else:
    # 3. Ridimensiona l'immagine per una migliore visualizzazione (opzionale)
    image = cv2.resize(image, (600, 400))

    # 4. Converti l'immagine a scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 5. Applica un filtro di sfocatura per ridurre il rumore
    blurred_image = cv2.medianBlur(gray_image, 5)

    # 6. Utilizza HoughCircles per rilevare i cerchi
    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=2.2,          # Risoluzione dell'accumulatore (riduci se non rileva cerchi)
        minDist=100,      # Distanza minima tra i centri dei cerchi
        param1=100,       # Parametro per il rilevamento dei bordi Canny
        param2=50,       # Soglia dell'accumulatore di Hough (riduci se non rileva cerchi)
        minRadius=10,    # Raggio minimo dei cerchi da rilevare
        maxRadius=80     # Raggio massimo dei cerchi da rilevare
    )

    # 7. Se ci sono cerchi rilevati, disegnarli sull'immagine originale
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)    # Disegna il cerchio
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)   # Disegna il centro

    # 8. Visualizza l'immagine originale con i cerchi rilevati usando Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Rilevamento dei Cerchi')
    plt.axis('off')
    plt.show()
