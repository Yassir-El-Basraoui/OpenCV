import cv2
import matplotlib.pyplot as plt

# 1. Carica l'immagine
img_path = 'OpenCV\image_2.png'  # Sostituisci con il percorso della tua immagine
image = cv2.imread(img_path)

# 2. Verifica che l'immagine sia stata caricata correttamente
if image is None:
    print("Errore nel caricamento dell'immagine.")
else:
    # 3. Converti l'immagine a scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Applica un filtro di sfocatura per ridurre il rumore (opzionale ma consigliato)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 5. Applica l'algoritmo di Canny per il rilevamento dei bordi
    # I valori di soglia più bassi e più alti regolano la sensibilità al rilevamento dei bordi
    edges = cv2.Canny(blurred_image, threshold1=100, threshold2=150)

    # 6. Visualizza l'immagine originale e l'immagine con i bordi rilevati
    plt.figure(figsize=(10, 5))

    # Mostra l'immagine originale
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Immagine Originale')
    plt.axis('off')

    # Mostra l'immagine con i bordi rilevati
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Bordi Rilevati con Canny')
    plt.axis('off')

    plt.show()
