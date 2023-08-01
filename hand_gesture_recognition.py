# Machine Learning
from joblib import load
# Computer Vision
import cv2
import mediapipe as mp
# Image Manipulation
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
# Text to Speech
import pygame
import threading
from gtts import gTTS
# Virtual Keyboard
from pynput.keyboard import Key, Controller
# Misc
import numpy as np
import os
import pandas as pd
import time
# DIY Algorithms
import common
"""
Configuration
"""
#Windows size
Width = 780
Height = 480
#Tamaño de imagen ayuda
h_ayuda = 60
w_ayuda = 60


"""
Función "say" Recibe como entrada la letra predicha para poder decir en audio que letra es
"""
def say(text):
    clock = pygame.time.Clock()

    #Genera un archivo de audio con el sonido de la letra
    tts = gTTS(text, lang='es', tld='com')
    tts.save("sound.mp3")

    #Carga el archivo de audio creado anteriormente
    pygame.mixer.init()
    pygame.mixer.music.load('sound.mp3')

    #Reproduce el archivo de audio
    pygame.mixer.music.play()

    #Verifica que hay un archivo en reproducción si es asi espera
    while pygame.mixer.music.get_busy():
       clock.tick(30)

    #Cierra la reproducción para poder eliminar el archivo
    pygame.mixer.quit()

    #Elimina el archivo de sonido
    os.remove("sound.mp3")

"""
LOAD MODEL
"""
path = os.path.join('models', 'hand_gesture_model_lda.joblib')
ldalassifier = load(path)

"""
CV2 CONFIGURATIONS
"""
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)

"""
MEDIAPIPE CONFIGURATIONS
"""
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

"""
PIL CONFIGURATIONS
"""
font_path = os.path.join("resourses", "FreeSans.otf")
print(font_path)
font = ImageFont.truetype(font_path, 50)

"""
VIRTUAL KEYBOARD
"""
keyboard = Controller()

"""
GET WIDTH AND HEIGHT OF IMAGE
"""
_, img = cap.read()
h, w, _ = img.shape

"""
SET VARIABLES AND CONSTANTS TO CALCULATE FPS
"""
prev_time = time.time()
# FPS update time in seconds
DISPLAY_TIME = 2
# Frame Count
fc = 0
# Frames per Second
fps = 0

"""
LETTER DETECTION VARIABLES
"""
current_character = ''
last_character = ''
# Time in seconds a person has to make sign before playing sound
DETECTION_TIME = 1
# Previous detection time
prev_detection_time = time.time()
# Flag to block detection
detection_mode = True

"""
INIT THREAD
"""
th = threading.Thread()

menu_fondo = cv2.imread("resourses/negro.png")
menu_fondo = cv2.resize(menu_fondo,(w,60))

menu_altura, menu_anchura, _ = menu_fondo.shape
ayuda = False
correcta = False
primer_ayuda = True

while True:

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    #si el usuario presiona la tecla "barra de espacio"
    if key == 32:
        if ayuda: 
            ayuda=False
        else:
            ayuda=True

    menu_fondo = cv2.imread("resourses/negro.png")
    menu_fondo = cv2.resize(menu_fondo,(w,60))

    _, img = cap.read()
    img = cv2.flip(img, 1)
    
    # Calculate FPS
    fc+=1
    time_diff = time.time() - prev_time
    if (time_diff) >= DISPLAY_TIME :
        fps = fc / (time_diff)
        fc = 0
        prev_time = time.time()
	
    # Add FPS count on frame
    fps_disp = f"FPS: {int(fps)}"
    cv2.putText(img, fps_disp, (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Añade el texto de exit
    cv2.putText(img, "Exit q", (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #convierte la imagen del video de BGR a RGB para luego enviarla aaaaaaaaaaaaaaaaaaa
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    hand_landmarks = common.get_right_hand_landmarks(results, h, w)
    

    #Se agrega la ayuda
    if ayuda and key>=97 and key<=122:
        primer_ayuda = False  
        if key == ord("j") or key == ord("k") or key == ord("q") or key == ord("x") or key == ord("z"):
            correcta = False
        else:
            img_ayuda = cv2.imread("resourses/ejemplos/"+chr(key)+".png")
            img_ayuda = cv2.resize(img_ayuda, (h_ayuda,w_ayuda))
            img_ayuda = cv2.flip(img_ayuda, 1)
            correcta = True

    if ayuda:
        if primer_ayuda:
            cv2.putText(menu_fondo, "Presione una letra para ver el ejemplo de la letra", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)     
        else:
            if correcta:
                cv2.putText(menu_fondo, "Recuerda que la imagen esta en espejo", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                menu_fondo[menu_altura-h_ayuda:menu_altura,menu_anchura-w_ayuda:menu_anchura] = img_ayuda
            else:
                cv2.putText(menu_fondo, "Esta letra no es valida presione otra letra", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
    # Añade el texto para la ayuda
        cv2.putText(menu_fondo, "Para activar la ayuda de letras, presione la barra de espacio", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if hand_landmarks is False:
        detection_mode = True
    if hand_landmarks is not False:
        # Draw Bounding Box
        df = pd.DataFrame(hand_landmarks)
        xmin = df[0].min()
        xmax = df[0].max()
        ymin = df[1].min()
        ymax = df[1].max()
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 3)

        # Normalize landmarks
        normalized_landmarks = common.normalize_hand_landmarks(hand_landmarks)
        df = pd.DataFrame(normalized_landmarks)
        df.columns = ['x', 'y', 'z']
        df.drop('z', axis=1, inplace=True)
        features = df.to_numpy().flatten()

        # Print landmarks normlized values
        for index, landmark in enumerate(hand_landmarks):
            cv2.putText(img, f"x: {df.iloc[index]['x']:.2}, y:{df.iloc[index]['y']:.2}", 
                (landmark[0], landmark[1] + 12), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)

        # Make predictions
        last_character = current_character
        current_character = ldalassifier.predict([features])
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 50),f'Prediction: {current_character[0]}',(255,0,255),font=font)
        img = np.asarray(img_pil)

        if (last_character != current_character):
            prev_detection_time = time.time()
            detection_mode = True
        # Simula el presionar una tecla
        # If current_character = last_character and time_diff > constant play sound
        if (last_character == current_character) \
                and (time.time() - prev_detection_time) > DETECTION_TIME \
                and detection_mode == True \
                and th.is_alive() == False:
            key = current_character[0]
            if key == 'null':
                continue
            th = threading.Thread(target=say, args=(key, ))
            th.start()
            keyboard.press(key)
            keyboard.release(key)
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
            detection_mode = False

    all = cv2.vconcat([img,menu_fondo])
    cv2.imshow("Image", all)


# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()