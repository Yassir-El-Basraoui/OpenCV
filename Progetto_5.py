import cv2
import matplotlib.pyplot as plt

# 1. Carica l'immagine
img_path = 'OpenCV\image_2.png'  # Sostituisci con il percorso della tua immagine
image = cv2.imread(img_path)

# 2. Verifica che l'immagine sia stata caricata correttamente
if image is None:
    print("Errore nel caricamento dell'immagine.")
else:
    # 3. Carica il classificatore di volti Haar Cascade
    cascade_path = 'haarcascade_frontalface_default.xml'  # Percorso del file .xml
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 4. Converti l'immagine a scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 5. Rileva i volti nell'immagine
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,  # Riduzione progressiva della dimensione dell'immagine
        minNeighbors=5,    # Minimo numero di vicini per ogni rettangolo
        minSize=(30, 30)   # Dimensione minima di un volto da rilevare
    )

    # 6. Disegna un rettangolo attorno a ogni volto rilevato
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blu per i rettangoli

    # 7. Visualizza l'immagine con i volti rilevati utilizzando Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Rilevamento dei Volti')
    plt.axis('off')
    plt.show()
