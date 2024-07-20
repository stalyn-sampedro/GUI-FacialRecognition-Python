#Librerías para la GUI
import customtkinter
import os
from PIL import Image, ImageTk
#Librerías para el Registro de Usuarios
import Funciones_RegistroDeUsuarios
#Librerías para el entrenamiento
import Funciones_EntrenarModelo
#Librerías para el reconocimiento
import Cámara
import Funciones_Reconocimiento
#
#Funciones

##Funciones para el Registro de Usuarios

##Funciones para el entrenamiento

##Funciones para el reconocimiento


#Importar funciones

#Ventana flotante
class ConfimaciónDeRegistroVentana(customtkinter.CTkToplevel):
    def __init__(self,Referencia):
        super().__init__()
        self.geometry("230x120")

        self.Mensaje_UsuarioRegistrado_VentanaConfirmación = customtkinter.CTkLabel(self, text="Usuario Registrado: "+Referencia.Registro_frame_Nombre_entry.get()+" "+Referencia.Registro_frame_Apellido_entry.get())
        self.Mensaje_UsuarioRegistrado_VentanaConfirmación.grid(row=0, column=0,padx=20, pady=20)
        #Botón Registrar Usuario 
        self.Confirmacion_Ventana_button_ConfirmarRegistrarUsuario = customtkinter.CTkButton(self, text="Continuar", command=lambda: self.LimpiarVentanaFrameReconocimiento(Referencia))
        self.Confirmacion_Ventana_button_ConfirmarRegistrarUsuario.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
    
    def LimpiarVentanaFrameReconocimiento(self,Referencia):
        print("xd")
        ##Entrada Nombre
        Referencia.Registro_frame_Nombre_entry.configure(placeholder_text="Ingrese su Nombre",state="normal")#Entrada del Nombre
        Referencia.Registro_frame_Nombre_entry.delete(0, "end")
        ##Entrada Apellido
        Referencia.Registro_frame_Apellido_entry.configure(placeholder_text="Ingrese su Apellido",state="normal")#Entrada del Apellido
        Referencia.Registro_frame_Apellido_entry.delete(0, "end")
        ##Botón Registrar Usuario 
        Referencia.Reconocimiento_frame_button_RegistrarUsuario.configure(state="normal")
        ##Label Cámara
        Referencia.Registro_frame_Cámara_label.configure(text="",image="")#Label De La imagen grande
        ##Label Cara Detectada
        Referencia.Registro_frame_CaraDetectada_label.configure(text="",image="")#Label De La imagen grande
        ##Label NumeroDeFotos
        Referencia.Registro_frame_NumeroDeFotos_label.configure(text="")#Label De La imagen grande
        ##Label Imagen Pose de Cara
        Referencia.Registro_frame_PoseCaraImagen_label.configure(text="",image="")#Label De La imagen grande
        ##Label Pose de Cara
        Referencia.Registro_frame_PoseCara_label.configure(text="")#Label De La imagen grande
        Cámara.DetenerCámara()
        self.destroy()

#Programa Principal
class App(customtkinter.CTk):#Clase para la GUI. Es una subclase de customkinter.CTK
    def __init__(self):#Se ejecuta cuando se construye un nuevo objeto

        super().__init__()#Llama al constructor de la clase customKindeter
        customtkinter.set_appearance_mode("Light")#Modo día
        
        self.ConfimaciónDeRegistro_Ventana=None

        self.title("Reconocimiento Facial con CNN")#Atributo título
        self.geometry("1080x720")#Atributo del tamaño del frame principal
        # Diseño de cuadrícula 1x2
        self.grid_rowconfigure(0, weight=1)#Índice 0 significa 1 fila
        self.grid_columnconfigure(1, weight=1)#Índice 1 significa 2 columnas
        # Carga las imágenes según el tema oscuro o claro
        self.DirectorioDeImagenes = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RecursosGráficos")
        self.logo_unach = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "UnachMovimiento.png")), size=(250, 118))#Contiene las imágenes PIL, no es una función
        self.large_test_image = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "cámaracerrada.png")), size=(720, 480))
        self.image_icon_image = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "image_icon_light.png")), size=(20, 20))
        self.cámara_image = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "cámara.png")), size=(20, 20))
  
        self.chat_image = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "chat_dark.png")), size=(20, 20))
   
        self.AgregarUsuario_image = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "add_user_dark.png")), size=(20, 20))
        self.ModeloCNN_image = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "CNN.png")), size=(20, 20))

        self.Configuración_image = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "settings.png")), size=(20, 20))
        
        #Crea el frame de navegación, a la izquierda
        self.Navegacion_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.Navegacion_frame.grid(row=0, column=0, sticky="nsew")#Un frame de 1x1
        self.Navegacion_frame.grid_rowconfigure(5, weight=1)#Lo cambia a 5x1 (El que está en la izquierda)
        #Label Imagen Unach
        self.Navegacion_frame_label_LogoUnachEnMovimiento = customtkinter.CTkLabel(self.Navegacion_frame, text="", image=self.logo_unach)
        self.Navegacion_frame_label_LogoUnachEnMovimiento.grid(row=0, column=0, padx=20, pady=10)
        #Botón de Reconocimiento
        self.Reconocimiento_frame_button = customtkinter.CTkButton(self.Navegacion_frame, corner_radius=0, height=40, border_spacing=10, text="Reconocimiento Facial",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),


                                                   image=self.cámara_image, anchor="w", command=self.Reconocimiento_frame_button_event)
        self.Reconocimiento_frame_button.grid(row=1, column=0, sticky="ew")
        #Botón de Agregar Usuario
        self.AgregarUsuario_frame_button = customtkinter.CTkButton(self.Navegacion_frame, corner_radius=0, height=40, border_spacing=10, text="Agregar Usuario",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.AgregarUsuario_image, anchor="w", command=self.AgregarUsuario_frame_button_event)
        self.AgregarUsuario_frame_button.grid(row=2, column=0, sticky="ew")
        #Botón de ModeloCNN
        self.ModeloCNN_frame_button = customtkinter.CTkButton(self.Navegacion_frame, corner_radius=0, height=40, border_spacing=10, text="Modelo CNN",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.ModeloCNN_image, anchor="w", command=self.ModeloCNN_frame_button_event)
        self.ModeloCNN_frame_button.grid(row=3, column=0, sticky="ew")
        #Botón de configuración
        self.Configuración_frame_button = customtkinter.CTkButton(self.Navegacion_frame, corner_radius=0, height=40, border_spacing=10, text="Configuración",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.Configuración_image, anchor="w", command=self.Configuración_frame_button_event)
        self.Configuración_frame_button.grid(row=4, column=0, sticky="ew")

        # Crea el Frame de Reconocimiento
        self.Reconocimiento_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.Reconocimiento_frame.grid_columnconfigure(0, weight=1)
        
        self.Reconocimiento_frame_Cámara_label = customtkinter.CTkLabel(self.Reconocimiento_frame, text="", image=self.large_test_image)#Label De La imagen grande
        self.Reconocimiento_frame_Cámara_label.grid(row=0, column=0, padx=20, pady=10)
        self.Estado_Cámara=False#True=Cámara encedida
        self.Reconocimiento_frame_button_AbrirCámara = customtkinter.CTkButton(self.Reconocimiento_frame, text="Abrir cámara", command=lambda: Funciones_Reconocimiento.DesiciónReconocimiento(self))
        self.Reconocimiento_frame_button_AbrirCámara.grid(row=1, column=0, padx=20, pady=10)
        
        # Crea el Frame de Registro
        self.Registro_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.Registro_frame.grid(row=0, column=2, sticky="nsew")#Un frame de 1x1
        #self.Registro_frame.grid_rowconfigure(0, weight=1)

        ##SubFrame para el ingreso de datos
        self.Registro_frame_Datos = customtkinter.CTkFrame(self.Registro_frame, corner_radius=0, fg_color="transparent")
        self.Registro_frame_Datos.grid(row=4, column=0, sticky="nsew")#Un frame de 1x1
        self.Registro_frame_Datos.grid_rowconfigure(4, weight=1)

        ##SubFrame para la muestra de video
        self.Registro_frame_Video = customtkinter.CTkFrame(self.Registro_frame, corner_radius=0, fg_color="transparent")
        self.Registro_frame_Video.grid(row=4, column=1, sticky="nsew")#Un frame de 1x1
        self.Registro_frame_Video.grid_rowconfigure(4, weight=1)

        ##SubFrame para la muestra del modelo
        self.Registro_frame_Modelo = customtkinter.CTkFrame(self.Registro_frame, corner_radius=0, fg_color="transparent")
        self.Registro_frame_Modelo.grid(row=4, column=2, sticky="nsew")#Un frame de 1x1
        self.Registro_frame_Modelo.grid_rowconfigure(4, weight=1)
        
        ##Label de Nombre y Apellido
        self.Registro_frame_NombreyApellido_label = customtkinter.CTkLabel(self.Registro_frame_Datos, text="Ingrese su Nombre y Apellido")#Label De La imagen grande
        self.Registro_frame_NombreyApellido_label.grid(row=0, column=0, sticky="ew")
        ##Entrada Nombre
        self.Registro_frame_Nombre_entry = customtkinter.CTkEntry(self.Registro_frame_Datos, placeholder_text="Ingrese su Nombre")#Entrada del Nombre
        self.Registro_frame_Nombre_entry.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        ##Entrada Apellido
        self.Registro_frame_Apellido_entry = customtkinter.CTkEntry(self.Registro_frame_Datos, placeholder_text="Ingrese su Apellido")#Entrada del Apellido
        self.Registro_frame_Apellido_entry.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        ##Botón Registrar Usuario 
        self.Reconocimiento_frame_button_RegistrarUsuario = customtkinter.CTkButton(self.Registro_frame_Datos, text="Empezar Registro", command=lambda: Funciones_RegistroDeUsuarios.RegistrarUsuario(self.Registro_frame_Nombre_entry.get(),self.Registro_frame_Apellido_entry.get(),self))
        self.Reconocimiento_frame_button_RegistrarUsuario.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        ##Label Cámara
        self.Registro_frame_Cámara_label = customtkinter.CTkLabel(self.Registro_frame_Video, text="")#Label De La imagen grande
        self.Registro_frame_Cámara_label.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        ##Label Cara Detectada
        self.Registro_frame_CaraDetectada_label = customtkinter.CTkLabel(self.Registro_frame_Video, text="")#Label De La imagen grande
        self.Registro_frame_CaraDetectada_label.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        ##Label NumeroDeFotos
        self.Registro_frame_NumeroDeFotos_label = customtkinter.CTkLabel(self.Registro_frame_Video, text="")#Label De La imagen grande
        self.Registro_frame_NumeroDeFotos_label.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        ##Label Imagen Pose de Cara
        self.Registro_frame_PoseCaraImagen_label = customtkinter.CTkLabel(self.Registro_frame_Modelo, text="")#Label De La imagen grande
        self.Registro_frame_PoseCaraImagen_label.grid(row=0, column=2, padx=20, pady=10)
        ##Label Pose de Cara
        self.Registro_frame_PoseCara_label = customtkinter.CTkLabel(self.Registro_frame_Modelo, text="")#Label De La imagen grande
        self.Registro_frame_PoseCara_label.grid(row=1, column=2, padx=20, pady=10)


        # Crea el Frame de ModeloCNN
        self.ModeloCNN_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        try:
            self.ModeloCNN_Imagen = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "modelo.png")), size=(250, 118))#Contiene las imágenes PIL, no es una función

            self.ModeloCNN_frame.grid(row=0, column=1, sticky="nsew")#Un frame de 1x1
            
            ##SubFrame para mostrar datos
            self.ModeloCNN_frame_Datos = customtkinter.CTkFrame(self.ModeloCNN_frame, corner_radius=0, fg_color="transparent")
            self.ModeloCNN_frame_Datos.grid(row=3, column=0, sticky="nsew")#Un frame de 1x1
            #self.ModeloCNN_frame_Datos.grid_rowconfigure(3, weight=1)

            ##SubFrame para mostrar el modelo
            self.ModeloCNN_frame_Modelo = customtkinter.CTkFrame(self.ModeloCNN_frame, corner_radius=0, fg_color="transparent")
            self.ModeloCNN_frame_Modelo.grid(row=3, column=1, sticky="nsew")#Un frame de 1x1
            #self.ModeloCNN_frame_Modelo.grid_rowconfigure(3, weight=1)

            ##Label Mensaje
            self.ModeloCNN_frame_Mensaje_label = customtkinter.CTkLabel(self.ModeloCNN_frame_Datos, text="Usuarios Registrados:")#Label De La imagen grande
            self.ModeloCNN_frame_Mensaje_label.grid(row=0, column=0, padx=20, pady=10)
            ##Combobox con usuarios
            self.ModeloCNN_frame_Usuarios_TextBox = customtkinter.CTkTextbox(self.ModeloCNN_frame_Datos)#Label De La imagen grande
            self.ModeloCNN_frame_Usuarios_TextBox.grid(row=1, column=0, padx=20, pady=10)
            ##Botón entrenar modelo
            self.ModeloCNN_frame_EntrenarModelo_button = customtkinter.CTkButton(self.ModeloCNN_frame_Datos, text="Entrenar Modelo", command=lambda: Funciones_EntrenarModelo.EntrenarCNN())
            self.ModeloCNN_frame_EntrenarModelo_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
            ##Label Imagen Modelo
            self.ModeloCNN_frame_ModeloImagen_label = customtkinter.CTkLabel(self.ModeloCNN_frame_Modelo, text="")#Label De La imagen grande
            self.ModeloCNN_frame_ModeloImagen_label.grid(row=0, column=1, padx=20, pady=10, sticky="ew")

            Funciones_EntrenarModelo.MostrarModeloEnFrame(self)

        except FileNotFoundError:#Si no hay un modelo creado

            self.ModeloCNN_frame.grid_columnconfigure(0, weight=1)#Un frame de 1x1
            ##Label Pose de Cara
            self.ModeloCNN_frame_Mensaje_label = customtkinter.CTkLabel(self.ModeloCNN_frame, text="No existe un modelo activo.")#Label De La imagen grande
            self.ModeloCNN_frame_Mensaje_label.grid(row=1, column=0, padx=20, pady=10)
            ##Botón Registrar Usuario 
            self.ModeloCNN_frame_EntrenarModelo_button = customtkinter.CTkButton(self.ModeloCNN_frame, text="Entrenar Modelo", command=lambda: Funciones_EntrenarModelo.EntrenarCNN())
            self.ModeloCNN_frame_EntrenarModelo_button.grid(row=2, column=0, padx=20, pady=10)

        # create third frame
        self.Configuración_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        # select default frame
        self.select_frame_by_name("Frame_Reconocimiento")

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.Reconocimiento_frame_button.configure(fg_color=("gray75", "gray25") if name == "Frame_Reconocimiento" else "transparent")
        self.Configuración_frame_button.configure(fg_color=("gray75", "gray25") if name == "Frame_Configuración" else "transparent")
        self.AgregarUsuario_frame_button.configure(fg_color=("gray75", "gray25") if name == "Frame_Registro" else "transparent")
        
        # show selected frame
        if name == "Frame_Reconocimiento":
            self.Reconocimiento_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.Reconocimiento_frame.grid_forget()
        if name == "Frame_Registro":
            self.Registro_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.Registro_frame.grid_forget()
        if name == "Frame_ModeloCNN":
            self.ModeloCNN_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.ModeloCNN_frame.grid_forget()
        if name == "Frame_Configuración":
            self.Configuración_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.Configuración_frame.grid_forget()

    def Reconocimiento_frame_button_event(self):
        self.select_frame_by_name("Frame_Reconocimiento")

    def AgregarUsuario_frame_button_event(self):
        self.select_frame_by_name("Frame_Registro")
    
    def ModeloCNN_frame_button_event(self):
        self.select_frame_by_name("Frame_ModeloCNN")
        self.DesiciónDeFrameModelo()

    def Configuración_frame_button_event(self):
        self.select_frame_by_name("Frame_Configuración")

    def Abrir_ConfimaciónDeRegistroVentana(self):
        if self.ConfimaciónDeRegistro_Ventana is None or not self.ConfimaciónDeRegistro_Ventana.winfo_exists():
                self.ConfimaciónDeRegistro_Ventana = ConfimaciónDeRegistroVentana(self)  # create window if its None or destroyed
                self.ConfimaciónDeRegistro_Ventana.focus()
        else:
                self.ConfimaciónDeRegistro_Ventana.focus()  # if window exists focus it

    def DesiciónDeFrameModelo(self):
        try:   
            self.ModeloCNN_Imagen = customtkinter.CTkImage(Image.open(os.path.join(self.DirectorioDeImagenes, "modelo.png")), size=(250, 118))#Contiene las imágenes PIL, no es una función
            
            self.ModeloCNN_frame.grid_forget()
            self.ModeloCNN_frame.grid(row=0, column=1, sticky="nsew")#Un frame de 1x1
            
            ##SubFrame para mostrar datos
            self.ModeloCNN_frame_Datos = customtkinter.CTkFrame(self.ModeloCNN_frame, corner_radius=0, fg_color="transparent")
            self.ModeloCNN_frame_Datos.grid(row=3, column=0, sticky="nsew")#Un frame de 1x1
            #self.ModeloCNN_frame_Datos.grid_rowconfigure(3, weight=1)

            ##SubFrame para mostrar el modelo
            self.ModeloCNN_frame_Modelo = customtkinter.CTkFrame(self.ModeloCNN_frame, corner_radius=0, fg_color="transparent")
            self.ModeloCNN_frame_Modelo.grid(row=3, column=1, sticky="nsew")#Un frame de 1x1
            #self.ModeloCNN_frame_Modelo.grid_rowconfigure(3, weight=1)

            ##Label Mensaje
            self.ModeloCNN_frame_Mensaje_label = customtkinter.CTkLabel(self.ModeloCNN_frame_Datos, text="Usuarios Registrados:")#Label De La imagen grande
            self.ModeloCNN_frame_Mensaje_label.grid(row=0, column=0, padx=20, pady=10)
            ##Combobox con usuarios
            self.ModeloCNN_frame_Usuarios_TextBox = customtkinter.CTkTextbox(self.ModeloCNN_frame_Datos)#Label De La imagen grande
            self.ModeloCNN_frame_Usuarios_TextBox.grid(row=1, column=0, padx=20, pady=10)
            ##Botón entrenar modelo
            self.ModeloCNN_frame_EntrenarModelo_button = customtkinter.CTkButton(self.ModeloCNN_frame_Datos, text="Entrenar Modelo", command=lambda: Funciones_EntrenarModelo.EntrenarCNN())
            self.ModeloCNN_frame_EntrenarModelo_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
            ##Label Imagen Modelo
            self.ModeloCNN_frame_ModeloImagen_label = customtkinter.CTkLabel(self.ModeloCNN_frame_Modelo, text="")#Label De La imagen grande
            self.ModeloCNN_frame_ModeloImagen_label.grid(row=0, column=1, padx=20, pady=10, sticky="ew")

            Funciones_EntrenarModelo.MostrarModeloEnFrame(self)

        except FileNotFoundError:#Si no hay un modelo creado
            self.ModeloCNN_frame.grid_columnconfigure(0, weight=1)#Un frame de 1x1
            ##Label Pose de Cara
            self.ModeloCNN_frame_Mensaje_label = customtkinter.CTkLabel(self.ModeloCNN_frame, text="No existe un modelo activo.")#Label De La imagen grande
            self.ModeloCNN_frame_Mensaje_label.grid(row=1, column=0, padx=20, pady=10)
            ##Botón Registrar Usuario 
            self.ModeloCNN_frame_EntrenarModelo_button = customtkinter.CTkButton(self.ModeloCNN_frame, text="Entrenar Modelo", command=lambda: Funciones_EntrenarModelo.EntrenarCNN())
            self.ModeloCNN_frame_EntrenarModelo_button.grid(row=2, column=0, padx=20, pady=10)

    

if __name__ == "__main__":
    app = App()#Objeto de la Clase
    app.mainloop()