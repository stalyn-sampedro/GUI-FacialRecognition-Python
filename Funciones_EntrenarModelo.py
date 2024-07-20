#Entrenamiento del modelo.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import gc

from tensorflow import keras
import customtkinter
from PIL import Image, ImageTk

def ObtenerDirectorios():
    DirectorioDeEntrenamiento=os.path.join(os.path.dirname(os.path.realpath(__file__)),"Fotos","Usuarios")
    #DirectorioDeValidacion=os.path.join(os.path.dirname(os.path.realpath(__file__)),"Fotos","Validacion")
    return DirectorioDeEntrenamiento

def EntrenarCNN(Ref,Referencia):
    #Variables
    Conv2D=tf.keras.layers.Conv2D
    MaxPooling2D=tf.keras.layers.MaxPooling2D
    Flatten=tf.keras.layers.Flatten
    Dense=tf.keras.layers.Dense
    Adam=tf.keras.optimizers.Adam
    LocallyConnected2D=tf.keras.layers.LocallyConnected2D
    #Parámetros para cargas los datos
    Lotes_tamaño=32 
    Imagen_ancho=224
    Imagen_alto=224
    PorcentajeDeValidacion=0.2
    epochs=20
    #Directorio de Usuarios
    DirectorioDeEntrenamiento=os.path.join(os.path.dirname(os.path.realpath(__file__)),"Fotos","Usuarios")
    #Crear conjunto de datos etiquetados
    Entrenamiento_ds=tf.keras.utils.image_dataset_from_directory(
        DirectorioDeEntrenamiento,
        validation_split=PorcentajeDeValidacion,
        subset='training',
        labels='inferred',#Se utiliza el nombre de la carpeta como etiqueta
        seed=321,
        image_size=(Imagen_ancho,Imagen_alto),
        color_mode='grayscale',
        batch_size=Lotes_tamaño)
    Validacion_ds=tf.keras.utils.image_dataset_from_directory(
        DirectorioDeEntrenamiento,
        validation_split=PorcentajeDeValidacion,
        subset='validation',
        labels='inferred',#Se utiliza el nombre de la carpeta como etiqueta
        seed=321,
        image_size=(Imagen_ancho,Imagen_alto),
        color_mode='grayscale',
        batch_size=Lotes_tamaño)
    NombresDeClases=Entrenamiento_ds.class_names #Recupera el nombre de las clases
    print(NombresDeClases)
    #Configurar para el rendimiento durante el entrenamiento.
    Autotune=tf.data.AUTOTUNE
    Entrenamiento_ds=Entrenamiento_ds.cache().shuffle(1000).prefetch(buffer_size=Autotune)
    Validacion_ds=Validacion_ds.cache().prefetch(buffer_size=Autotune)
    #Cache se usa para mantener imagenes en la memoria luego de cargarse fuera del disco. Evita cuellos de botella.
    # Superpone le procesamietno de datos y la ejecución del modelo durante el entrenamiento
    #
    #Aumento de Datos
    Data_Augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomBrightness(factor=0.2,input_shape=(Imagen_ancho,Imagen_alto,1)),
            tf.keras.layers.RandomContrast(factor=0.2,input_shape=(Imagen_ancho,Imagen_alto,1))
        ]
    )
    #
    #Crear el modelo
    NúmeroDeClases=len(NombresDeClases)

    Modelo = tf.keras.Sequential([
        Data_Augmentation,
        tf.keras.layers.Rescaling(1./255),
        Conv2D(16,11,padding='same', activation='relu',input_shape=(224,224,1)),
        MaxPooling2D(2),
        Conv2D(128,9,padding='same',activation='relu'),
        Conv2D(128,9,padding='same',activation='relu'),
        tf.keras.layers.Dropout(0.2),
        MaxPooling2D(2),
        Conv2D(256,5,padding='same',activation='relu'),
        Conv2D(256,5,padding='same',activation='relu'),
        Conv2D(256,5,padding='same',activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        Dense(NúmeroDeClases, activation='softmax')
    ])
    
    #Compilar Modelo
    Modelo.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    History= Modelo.fit(
        Entrenamiento_ds,
        validation_data=Validacion_ds,
        epochs=epochs
    ) 

    #Guardar el modelo para predicciones futuras
    Modelo.save('ModeloDeReconocimientoFacial.h5')

    #Resultados de entrenamiento
    acc = History.history['accuracy']
    val_acc = History.history['val_accuracy']

    loss = History.history['loss']
    val_loss = History.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    ArchivoImagenModelo=os.path.join(os.path.dirname(os.path.realpath(__file__)),"RecursosGráficos","modelo.png")
    plt.savefig(ArchivoImagenModelo)

    #Guardar Lista de Usuarios
    UsuariosRegistrados_Archivo="UsuariosRegistrados.txt"
    with open(UsuariosRegistrados_Archivo, "w") as Archivo:
        # Escribe cada elemento de la lista en una línea separada
        for NombreActual in NombresDeClases:
            Archivo.write(NombreActual + "\n")

    for widget in Ref.ModeloCNN_frame.winfo_children():
            widget.destroy()
    Ref.CargarTabModeloCNN(Ref,Referencia)

def MostrarModeloEnFrame(self):
    ArchivoUsuarios=os.path.join(os.path.dirname(os.path.realpath(__file__)),"UsuariosRegistrados.txt")
    DirectorioDeImagenes = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RecursosGráficos")

    #Carga el nombre de clases:
    ListaDeUsuarios=[]

    with open(ArchivoUsuarios, "r") as Archivo:
        # Lee cada línea del archivo y agrega su contenido a la lista
        for NombreActual in Archivo:
            ListaDeUsuarios.append(NombreActual.strip())  # strip() elimina los caracteres de nueva línea (\n)

    #Leer el archivo y cargar la imagen
    for NombreActual in ListaDeUsuarios:
        self.ModeloCNN_frame_Usuarios_TextBox.insert("end", NombreActual + "\n")
    
    self.ModeloCNN_frame_Usuarios_TextBox.configure(state="disabled")

    ImagenModelo=customtkinter.CTkImage(Image.open(os.path.join(DirectorioDeImagenes, "modelo.png")), size=(500, 500))
    self.ModeloCNN_frame_ModeloImagen_label.configure(image=ImagenModelo)




