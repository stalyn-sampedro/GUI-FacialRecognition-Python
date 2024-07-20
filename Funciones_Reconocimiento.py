#Utiliza el modelo guardado y realiza una predicción.

import tensorflow as tf
import numpy as np
import os

import cv2 #Importa la Librería de OpenCV, usada para manipular las imágenes

import datetime

#Para la detección
import imutils #Realiza cambios de tamaño en las imagenes
from mtcnn.mtcnn import MTCNN #RedNeuronal Convolucional entrenada para reconocer rostros
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Importar Funciones
import Cámara
import Funciones_RegistroDeUsuarios
import Funciones_mysql


def EmpezarReconocimiento(self):
    self.Reconocimiento_frame_button_AbrirCámara.configure(text="Detener Cámara")

    RutaDelModelo=os.path.join(os.path.dirname(os.path.realpath(__file__)),"ModeloDeReconocimientoFacial3.h5")

    Modelo=tf.keras.models.load_model(RutaDelModelo)
    Modelo.summary()
    #
    #Nombre de clases:
    ListaDeUsuarios=[]
    ArchivoUsuarios=os.path.join(os.path.dirname(os.path.realpath(__file__)),"UsuariosRegistrados.txt")

    with open(ArchivoUsuarios, "r") as Archivo:
        # Lee cada línea del archivo y agrega su contenido a la lista
        for NombreActual in Archivo:
            ListaDeUsuarios.append(NombreActual.strip())  # strip() elimina los caracteres de nueva línea (\n)

    #Recorte del rostro en la imagen
    AnchoDeImagen,AltoDeImagen=224, 224 #Para normalizar las imagenes
    #
    #Iniciar MTCNN
    detector = MTCNN()#Red neuronal Convolucional, se le asigna a una variable.
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  
    #
    #Iniciar Cámara
    Cámara.ConfigurarCámara()

    PrediccionTemporal=""

    BucleDeReconocimiento(self,detector,AnchoDeImagen,AltoDeImagen,Modelo,ListaDeUsuarios,PrediccionTemporal)

def BucleDeReconocimiento(self,detector,AnchoDeImagen,AltoDeImagen,Modelo,ListaDeUsuarios,PrediccionTemporal):
    if self.Estado_Cámara:
        Frame=Cámara.ObtenerFrameActualDeLaCámara()
        if Frame is not None:
            Caras = detector.detect_faces(Frame)#Los objetos JSON con las caras detectadas se pasan a la variable Caras
            for i in range(len(Caras)):#Revisa todas las caras en el frame.
                ProbabilidadDeCara=Caras[i]['confidence']#Guardo la probailidad de que el objeto actualsea una cara.
                if ProbabilidadDeCara >= 0.95:#Si hay una probabilidad del 95% mínimo, guarda la cara.
                    x1,y1,AnchoDeLaCajaDelimitadora,AltoDeLaCajaDelimitadora= Caras[i]['box']#Guardo los puntos x,y y el ancho y alto de la caja delimitadora de las caras
                    x2,y2=x1+AnchoDeLaCajaDelimitadora,y1+AltoDeLaCajaDelimitadora
                    CaraDetectadaAColor=Frame[y1:y2, x1:x2]# Se recorta la región de la caja delimitadora.
                    CaraDetectadaABlancoYNegro=cv2.cvtColor(CaraDetectadaAColor, cv2.COLOR_BGR2GRAY)#Pasa la cara a blanco y negro.
                    CaraRedimensionada=cv2.resize(CaraDetectadaABlancoYNegro,(AnchoDeImagen,AltoDeImagen))#Normaliza las imágenes.
                    
                    #Cara Detectada:
                    cv2.rectangle(Frame,(x1, y1), (x2, y2),(255,0,0),2)
                    # Area de la cara
                    Base = x2 - x1
                    Altura = y2 - y1
                    Area=Base*Altura

                    #Predicción:
                    # Preparar imagen
                    ImagenDePredicción_array=CaraRedimensionada#/255.0

                    ImagenDePredicción_array=tf.expand_dims(ImagenDePredicción_array,0)#Convierte la imagen en un tensor
                    #
                    Predicciones=Modelo.predict(ImagenDePredicción_array)
                    #score=tf.nn.softmax(Predicciones)

                    ClasePredicha=ListaDeUsuarios[np.argmax(Predicciones)]
                    Confianza=100 * np.max(Predicciones)
                    Confianza=round(Confianza,3)
                    #and Area >= 427989
                    if Confianza>=95:
                        print(
                            "Esta imagen más seguramente pertenece a {} con un {:.2f} por ciento de confianza.".format(ClasePredicha, Confianza)
                        )
                        Funciones_mysql.RegistrarReconocimiento(ClasePredicha,Confianza,self,CaraRedimensionada)

                        cv2.putText(Frame, f"{ClasePredicha} ({Confianza:.2f}%)", (x1, y1 + AltoDeLaCajaDelimitadora + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    else:
                        cv2.putText(Frame, "Desconocido", (x1, y1 + AltoDeLaCajaDelimitadora + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
            Cámara.MostrarFrameEnEtiqueta(self,Frame,(640, 360),self.Reconocimiento_frame_Cámara_label)#Mostrar video
        self.after(50,lambda: BucleDeReconocimiento(self,detector,AnchoDeImagen,AltoDeImagen,Modelo,ListaDeUsuarios,PrediccionTemporal))



def DesiciónReconocimiento(self):
    self.Estado_Cámara = not self.Estado_Cámara
    if self.Estado_Cámara:
        EmpezarReconocimiento(self)
        self.Reconocimiento_frame_button_AbrirCámara.configure(text="Detener Cámara")
    else:
        self.Reconocimiento_frame_button_AbrirCámara.configure(text="Iniciar Cámara")
        self.Reconocimiento_frame_Cámara_label.configure(image=self.large_test_image)
        self.Reconocimiento_frame_CaraDetectada_label.configure(text="",image="")
        self.Reconocimientoframe_UsuarioDetectado_label.configure(text="")
        
        Cámara.DetenerCámara()







