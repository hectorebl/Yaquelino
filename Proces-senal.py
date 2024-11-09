import os
import numpy as np
import pandas as pd  # Importa pandas para manejar tablas
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# Configuración de filtros
fs = 1000  # Frecuencia de muestreo estimada en Hz
lowcut = 20   # Frecuencia de corte inferior para el filtro paso banda
highcut = 450 # Frecuencia de corte superior para el filtro paso banda

# Ruta de la carpeta con los archivos de texto
folder_path = 'C:\\Users\\hecto\\OneDrive\\Documentos\\EMGs\\'  # Cambia esta ruta según tu carpeta

# Función para aplicar un filtro paso banda
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Función de suavizado (media móvil)
def moving_average(signal, window_size):
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='valid')
    return smoothed_signal

# Cálculo de RMS en ventanas deslizantes
def calculate_rms(emg_signal, window_size):
    squared_signal = np.square(emg_signal)
    window = np.ones(window_size) / window_size
    rms_signal = np.sqrt(np.convolve(squared_signal, window, mode='valid'))
    return rms_signal

# Procesar cada archivo de texto en la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        print(f"Procesando archivo: {filename}")
        
        # Cargar la señal EMG desde el archivo .txt
        data = np.loadtxt(file_path, comments='#', delimiter='\t', usecols=(5,))  # Columna A1 (índice 5)
        
        # Aplicación de los filtros a la señal EMG
        filtered_emg = bandpass_filter(data, lowcut, highcut, fs)
        
        # Rectificación completa de la señal EMG
        rectified_emg = np.abs(filtered_emg)
        
        # Crear una tabla con la señal EMG rectificada
        df_rectified = pd.DataFrame({"Rectified EMG": rectified_emg})
        
        # Guardar la tabla como un archivo .csv
        csv_output_path = os.path.join(folder_path, f"{filename}_rectified.csv")
        df_rectified.to_csv(csv_output_path, index=False)
        print(f"Tabla de la señal rectificada guardada en: {csv_output_path}")
        
        # Suavizado mediante media móvil
        window_size = 200
        smoothed_emg = moving_average(rectified_emg, window_size)
        
        # Cálculo de RMS
        rms_emg = calculate_rms(smoothed_emg, window_size)
        
        # Cálculo de la media y la integración de la señal EMG rectificada
        mean_emg = np.mean(rectified_emg)
        iemg = np.sum(rectified_emg)
        
        # Calcular la mediana de frecuencia con el método de Welch
        frequencies, power_spectrum = welch(rectified_emg, fs, nperseg=1024)
        median_freq = np.median(frequencies)
        
        # Imprimir resultados clave
        print("Media de la señal EMG rectificada:", mean_emg)
        print("Integración de la señal EMG (iEMG):", iemg)
        print("Mediana de frecuencia:", median_freq)
        
        # Graficar la señal EMG procesada (opcional)
        plt.figure(figsize=(12, 6))
        plt.plot(rms_emg, color='red')
        plt.title(f"RMS of EMG Signal for {filename}")
        plt.xlabel("Sample Window")
        plt.ylabel("RMS Amplitude")
        plt.grid(True)
        plt.show()
        
        # Graficar el espectro de frecuencia (opcional)
        plt.figure(figsize=(12, 6))
        plt.semilogy(frequencies, power_spectrum, color='blue')
        plt.title(f"Frequency Spectrum of EMG Signal (Welch Method) for {filename}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectrum")
        plt.grid(True)
        plt.show()
