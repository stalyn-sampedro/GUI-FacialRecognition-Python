#Librerías

import cv2 #Importa la Librería de OpenCV, usada para manipular las imágenes
import os # Crea Directorios
import tensorflow as tf

#Para la cámara:
import Cámara
from PIL import Image, ImageTk
import customtkinter
#Para la detección
import imutils #Realiza cambios de tamaño en las imagenes
from mtcnn.mtcnn import MTCNN #RedNeuronal Convolucional entrenada para reconocer rostros
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#Clase Padre

def CrearCarpetaDeUsuario(Nombre, Apellido):

    Directorio_Usuarios=os.path.join("Fotos","Usuarios")
    Directorio_Raiz=os.path.join(os.path.dirname(os.path.realpath(__file__)),Directorio_Usuarios)
    UsuarioActual=Nombre+"_"+Apellido
    Directorio_UsuarioActural=os.path.join(Directorio_Raiz,UsuarioActual)

    if not os.path.exists (Directorio_UsuarioActural): #Si no existe la carpeta especificada
        print('Carpeta creada:', Directorio_UsuarioActural)
        os.makedirs(Directorio_UsuarioActural) #La crea

    return Directorio_UsuarioActural

def Prueba():
    print(self.Nombre)

def RegistrarUsuario(Nombre,Apellido,self,Referencia):

    #Variables
    Directorio_UsuarioActural=CrearCarpetaDeUsuario(Nombre,Apellido)
    DirectorioDeImagenes = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RecursosGráficos")
    Contador=1
    AnchoDeImagen=224
    AltoDeImagen=224 #Para normalizar las imagenes
    TotalDeImagenes=600

    #Desactivar Botón y Entradas Mientras se realiza el proceso
    self.Registro_frame_Nombre_entry.configure(state="disabled")
    self.Registro_frame_Apellido_entry.configure(state="disabled")
    self.Reconocimiento_frame_button_RegistrarUsuario.configure(state="disabled")
    #

    #Iniciar MTCNN
    detector = MTCNN()#Red neuronal Convolucional, se le asigna a una variable.
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  
    #
    #Iniciar Cámara
    Cámara.ConfigurarCámara()

    BuclDeRegistro(Contador,AnchoDeImagen,AltoDeImagen,TotalDeImagenes,DirectorioDeImagenes,Directorio_UsuarioActural,detector,Nombre,Apellido,self,Referencia)





def BuclDeRegistro(Contador,AnchoDeImagen,AltoDeImagen,TotalDeImagenes,DirectorioDeImagenes,Directorio_UsuarioActural,detector,Nombre,Apellido,self,Referencia):
    
    if Contador == 1:
        self.Registro_frame_PoseCara_label.configure(text="Mire hace al frente")
        ImagenReferencia=customtkinter.CTkImage(Image.open(os.path.join(DirectorioDeImagenes, "frente.png")), size=(250, 250))
        self.Registro_frame_PoseCaraImagen_label.configure(image=ImagenReferencia)

    elif Contador ==120:
        self.Registro_frame_PoseCara_label.configure(text="Mire hace la derecha")
        ImagenReferencia=customtkinter.CTkImage(Image.open(os.path.join(DirectorioDeImagenes, "izquierda.png")), size=(250, 250))
        self.Registro_frame_PoseCaraImagen_label.configure(image=ImagenReferencia)

    elif Contador ==240:
        self.Registro_frame_PoseCara_label.configure(text="Mire hace la izquierda")
        ImagenReferencia=customtkinter.CTkImage(Image.open(os.path.join(DirectorioDeImagenes, "derecha.png")), size=(250, 250))
        self.Registro_frame_PoseCaraImagen_label.configure(image=ImagenReferencia)

    elif Contador ==360:
        self.Registro_frame_PoseCara_label.configure(text="Mire hacia arriba")
        ImagenReferencia=customtkinter.CTkImage(Image.open(os.path.join(DirectorioDeImagenes, "arriba.png")), size=(250, 250))
        self.Registro_frame_PoseCaraImagen_label.configure(image=ImagenReferencia)

    elif Contador ==480:
        self.Registro_frame_PoseCara_label.configure(text="Mire hacia abajo")
        ImagenReferencia=customtkinter.CTkImage(Image.open(os.path.join(DirectorioDeImagenes, "abajo.png")), size=(250, 250))
        self.Registro_frame_PoseCaraImagen_label.configure(image=ImagenReferencia)

    Frame=Cámara.ObtenerFrameActualDeLaCámara()
    if Frame is not None:
        Cámara.MostrarFrameEnEtiqueta(self,Frame,(640, 360),self.Registro_frame_Cámara_label)#Mostrar video
        CajaDelimitadora=DetecciónConMTCNN(detector,Frame)

        try:
            x1,x2,y1,y2=CajaDelimitadora
    
            CaraDetectadaAColor=Frame[y1:y2, x1:x2]# Se recorta la región de la caja delimitadora.
            CaraDetectadaABlancoYNegro=cv2.cvtColor(CaraDetectadaAColor, cv2.COLOR_BGR2GRAY)#Pasa la cara a blanco y negro.
            CaraRedimensionada=cv2.resize(CaraDetectadaABlancoYNegro,(AnchoDeImagen,AltoDeImagen))#Normaliza las imágenes.

            NombreDelArchivo = Nombre+Apellido+"_{}.png".format(Contador)#Nombre para el archivo de la foto con el número de foto al final.
            RutaDelArchivo = os.path.join(Directorio_UsuarioActural,NombreDelArchivo)      

            cv2.imwrite (RutaDelArchivo,CaraRedimensionada)#Guarda la foto
            ImagenDetectada=customtkinter.CTkImage(Image.open(os.path.join(DirectorioDeImagenes, "abajo.png")), size=(250, 250))
            
            Cámara.MostrarFrameEnEtiqueta(self,CaraRedimensionada,(224, 224),self.Registro_frame_CaraDetectada_label)#Mostrar video
            self.Registro_frame_NumeroDeFotos_label.configure(text="Número de fotos registradas: " + str(Contador))
            #
            Contador=Contador + 1 #Cuenta una imagen guardada.
        except TypeError:
            print("No Hay Caras detectadas")

    if Contador != 601:
        self.after(50, lambda: BuclDeRegistro(Contador,AnchoDeImagen,AltoDeImagen,TotalDeImagenes,DirectorioDeImagenes,Directorio_UsuarioActural,detector,Nombre,Apellido,self,Referencia))
    else:
        for widget in self.ModeloCNN_frame.winfo_children():
            widget.destroy()
        for widget in self.EliminarUsuarios_frame.winfo_children():
            widget.destroy()
        
        self.CargarTabEliminarUsuarios(self)
        self.CargarTabModeloCNN(self, Referencia)
        Referencia.Abrir_ConfimaciónDeRegistroVentana(self)
        
def DetecciónConMTCNN(detector,ImagenRegistro):
    
    Caras = detector.detect_faces(ImagenRegistro)#Los objetos JSON con las caras detectadas se pasan a la variable Caras
    
    if len(Caras)!=0:#Si existe una cara
            Caras=sorted(Caras,key=lambda x: x['confidence'],reverse=True)#Ordena De Mayor a Menor
            ProbabilidadDeCara=Caras[0]['confidence']#Guardo la probailidad de que el objeto actualsea una cara.
            print(ProbabilidadDeCara)
            if ProbabilidadDeCara >= 0.95:#Si hay una probabilidad del 90% mínimo, guarda la cara.
                x1,y1,AnchoDeLaCajaDelimitadora,AltoDeLaCajaDelimitadora= Caras[0]['box']#Guardo los puntos x,y y el ancho y alto de la caja delimitadora de las caras
                x2,y2=x1+AnchoDeLaCajaDelimitadora,y1+AltoDeLaCajaDelimitadora
                CajaDelimitadora=(x1,x2,y1,y2)
                return CajaDelimitadora

