# Proyecto Manómetro

Un sistema avanzado de IoT e Inteligencia Artificial diseñado para leer manómetros analógicos en tiempo real utilizando dispositivos Edge (NVIDIA Jetson). Transmite la telemetría y el video a un Dashboard centralizado alojado en la nube (AWS Elastic Beanstalk).

## 🌟 Características Principales

### 🖥️ Edge Computing (NVIDIA Jetson)
- **Visión Computacional Acelerada:** Utiliza un modelo **YOLOv8 de Segmentación** compilado con **TensorRT** y acelerado mediante CUDA para detectar las partes clave del manómetro (aguja, centro, y valores impresos) a alta velocidad.
- **Calibración Automática con OCR:** Durante los primeros segundos, el sistema utiliza **PyTesseract** para leer de forma autónoma los números impresos en el manómetro físico (valor máximo y mínimo).
- **Cálculo Trigonométrico de Presión:** Transforma el ángulo detectado de la aguja en un valor preciso de presión mediante cálculos geométricos.
- **Sistema de Alertas (Resend API):** Envía automáticamente un correo electrónico de emergencia adjuntando una fotografía en caso de que la presión supere el umbral de peligro.
- **Servidor Local de Contingencia:** Levanta un servidor Flask local en el puerto `5000` para transmitir video en caso de pérdida de conexión con la nube.

### ☁️ Cloud & Dashboard (AWS)
- **AWS IoT Greengrass & Core:** La Jetson se comunica de manera bidireccional mediante el protocolo ligero MQTT. Publica las lecturas de presión y se suscribe a comandos en vivo del dashboard.
- **AWS S3:** Almacena el *stream* visual del manómetro (fotografías analizadas con la IA).
- **Dashboard en Elastic Beanstalk:** Una plataforma web desarrollada en Flask/Python que permite visualizar el video casi en tiempo real y enviar comandos de control a la Jetson (Pausar/Arrancar).
- **Autenticación Segura:** El acceso al Dashboard web está protegido mediante **Amazon Cognito**.

## 🛠️ Tecnologías Utilizadas
* **Lenguajes:** Python 3, JavaScript, HTML5, CSS3, Bash, PowerShell.
* **Inteligencia Artificial:** YOLOv8 (Ultralytics), NVIDIA TensorRT, CUDA, OpenCV, Tesseract OCR.
* **AWS Services:** IoT Core, Greengrass V2, S3, Elastic Beanstalk, Cognito.

---
*Desarrollado para la monitorización remota, autónoma e inteligente de equipamiento industrial crítico.*
