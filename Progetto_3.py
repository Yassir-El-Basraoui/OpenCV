import cv2
import matplotlib.pyplot as plt

# 1. Carica l'immagine
img_path = 'image.png'  # Sostituisci con il percorso della tua immagine
image = cv2.imread(img_path)

# 2. Verifica che l'immagine sia stata caricata correttamente
if image is None:
    print("Errore nel caricamento dell'immagine.")
else:
    # 3. Converti l'immagine a scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Applica la soglia per separare gli oggetti dallo sfondo
    _, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # 5. Trova i contorni
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Disegna i contorni sull'immagine originale
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)  # Verde per i contorni, spessore 2

    # 7. Visualizza le immagini utilizzando Matplotlib
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Immagine Originale')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(threshold_image, cmap='gray')
    plt.title('Immagine con Soglia')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
    plt.title('Immagine con Contorni')
    plt.axis('off')

    plt.show()
