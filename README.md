# Dactilología en lengua de señas mexicana usando visión por computadora  y máquinas de soporte vectorial

## Instalación
Se sugiere ejecutar el código en una máquina Linux con anaconda instalado, para instalar las librerías es necesario ejecutar los siguientes comandos en el orden en que se muestran:

```
conda create -n hand-gesture
conda activate hand-gesture
conda install -c anaconda scikit-learn joblib ipykernel
conda install -c conda-forge notebook pandas pynput
pip install pygame gTTS mediapipe opencv-python
```

Posteriormente es necesario generar los modelos de clasificación ejecutando el siguiente comando:
```
python create_models.py
```

Una vez creados los modelos es posible crear el script ejecutando el siguiente comando:
```
hand_gesture_recognition.py
```
