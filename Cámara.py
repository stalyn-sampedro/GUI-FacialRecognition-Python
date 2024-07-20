import cv2
from PIL import Image, ImageTk
import customtkinter
import threading


def ConfigurarCámara():
    global latest_frame
    global cap
    global DetenerHilo
    global frame_thread

    latest_frame = None  # Variable para almacenar el último fotograma
    cap= None
    DetenerHilo=False


    # Inicia el hilo para la lectura de fotogramas
    frame_thread = threading.Thread(target=read_camera_frames)
    frame_thread.daemon = True  # El hilo se ejecutará como demonio(se cerrará cuando el programa principal termine)
    frame_thread.start()

def read_camera_frames():
    global latest_frame
    global cap
    global DetenerHilo
    url = "rtsp://admin:FMLAGI@192.168.100.77:554/" #Dirección del stream

    cap = cv2.VideoCapture(url)  # Reemplaza la URL con la de tu cámara

    while not DetenerHilo:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame  # Actualiza el último fotograma

    cap.release()

def ObtenerFrameActualDeLaCámara():
    global latest_frame
    if latest_frame is not None:
        return(latest_frame)

def DetenerCámara():
    global DetenerHilo
    global frame_thread

    DetenerHilo=True
    frame_thread.join()

def MostrarFrameEnEtiqueta(self,Frame, Tamaño, Etiqueta):
    if Etiqueta is not None and Frame is not None:
        ImagenOpenCV=cv2.cvtColor(Frame, cv2.COLOR_BGR2RGBA)
        ImagenCapturada=Image.fromarray(ImagenOpenCV)
        FrameEnLaEtiqueta=customtkinter.CTkImage(ImagenCapturada, size=Tamaño)
        Etiqueta.configure(image=FrameEnLaEtiqueta)
        
def MostrarCámara(self,Etiqueta):
    MostrarFrameEnEtiqueta(self,ObtenerFrameActualDeLaCámara(),(720, 480),Etiqueta)
    self.after(5, lambda: MostrarCámara(self,Etiqueta))
    

def Iniciar_Detener_Cámara_Reconocimiento(self,Etiqueta):

        self.Estado_Cámara = not self.Estado_Cámara
        if self.Estado_Cámara:
            ConfigurarCámara()
            MostrarCámara(self,Etiqueta)
            self.Reconocimiento_frame_button_AbrirCámara.configure(text="Detener Cámara")
        else:
            # Detén la actualización del video
            global cap
            if cap is not None:
                cap.release()  # Detener la captura de la cámara
            self.Reconocimiento_frame_button_AbrirCámara.configure(text="Iniciar Cámara")
            Etiqueta.configure(image=self.large_test_image)
        # Cambia el estado de video_actualizando