# Detección de Objetos con YOLO

Este repositorio contiene los scripts necesarios para ejecutar el modelo de detección de objetos YOLO. Siga las instrucciones a continuación para configurar el entorno y ejecutar la aplicación.

## Prerrequisitos

Antes de comenzar, asegúrese de tener Anaconda (o Miniconda) instalado en su computadora.
* [Guía de instalación de Anaconda](https://docs.anaconda.com/free/anaconda/install/)

## 🚀 Instalación y Ejecución

Siga los pasos a continuación en su terminal para ejecutar el modelo:

### 1. Acceda al directorio del proyecto
Reemplace `<PATH>` con la ruta real donde se encuentran los archivos del proyecto:
```bash
cd <PATH>
```

### 2. Cree el entorno virtual (si es la primera vez)
Se recomienda crear un entorno aislado para evitar conflictos de dependencias:
```bash
conda create --name yolo-env1 python=3.9
```

### 3. Active el entorno virtual
```bash
conda activate yolo-env1
```

### 4. Instale las dependencias
Con el entorno activado, instale la biblioteca oficial de YOLO:
```bash
pip install ultralytics
```

### 5. Ejecute el modelo
Por último, ejecute el script de detección:
```bash
python yolo_detect.py --model bestAnalogBoundingBox.pt --source usb0 --resolution 1280x720
```

### 6. Cerrar el modelo
```bash
Presione 'q' para cerrar la cámara
```

## 📚 Créditos y Referencias

El código utilizado en `yolo_detect.py` fue basado y adaptado del siguiente tutorial:
* https://youtu.be/r0RspiLG260?si=vngWBlRWqf2YWezb 
