import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def autocorrelation_method(signal, frame_length, sampling_rate):
    """
    Линейное предсказание анализа речевого сигнала с использованием
    автокорреляционного метода.

    Аргументы:
    signal: numpy массив, входной речевой сигнал
    frame_length: длительность фрейма в миллисекундах
    sampling_rate: частота дискретизации сигнала в Гц

    Возвращает:
    a_coeffs: numpy массив, коэффициенты анализа
    """
    frame_size = int((frame_length / 1000) * sampling_rate)
    num_frames = len(signal) // frame_size

    a_coeffs = []

    for i in range(num_frames):
        frame = signal[i * frame_size: (i + 1) * frame_size]
        frame = np.expand_dims(frame, axis=1)  # Преобразование в столбцовый вектор
        autocorr = np.correlate(frame[:, 0], frame[:, 0], mode='full')
        r = autocorr[frame_size - 1:]  # Autocorrelation values
        r = r[:frame_size]  # Retain only the values for positive lags
        # Solve the normal equations to find the LPC coefficients
        a = np.dot(np.linalg.inv(np.dot(frame.T, frame)), np.dot(frame.T, r))
        a_coeffs.append(a)

    return np.array(a_coeffs)

def lpc_decoder(a_coeffs, frame_length, sampling_rate):
    """
    LPC-декодер для восстановления сигнала.

    Аргументы:
    a_coeffs: numpy массив, коэффициенты LPC
    frame_length: длительность фрейма в миллисекундах
    sampling_rate: частота дискретизации сигнала в Гц

    Возвращает:
    reconstructed_signal: numpy массив, восстановленный сигнал
    """
    frame_size = int((frame_length / 1000) * sampling_rate)
    reconstructed_signal = np.zeros(len(a_coeffs) * frame_size)

    for i, a in enumerate(a_coeffs):
        # Формируем импульсную характеристику из коэффициентов LPC
        impulse_response = np.zeros(frame_size)
        impulse_response[0] = 1
        reconstructed_frame = lfilter([1], np.hstack(([1], -a)), impulse_response)
        reconstructed_signal[i * frame_size: (i + 1) * frame_size] = reconstructed_frame[:frame_size]

    return reconstructed_signal

# Пример использования:
# Замените sample_signal на ваш речевой сигнал
# Замените значения frame_length и sampling_rate на соответствующие вашим данным
sample_signal = np.random.rand(10000)  # Пример случайного сигнала
frame_length = 20  # мс
sampling_rate = 44100  # Гц

# Анализ сигнала
a_coeffs = autocorrelation_method(sample_signal, frame_length, sampling_rate)

# Восстановление сигнала
reconstructed_signal = lpc_decoder(a_coeffs, frame_length, sampling_rate)

# Вывод спектрограмм исходного и восстановленного сигналов
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.specgram(sample_signal, Fs=sampling_rate, NFFT=256, cmap='viridis')
plt.title('Спектрограмма исходного сигнала')
plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц)')

plt.subplot(1, 2, 2)
plt.specgram(reconstructed_signal, Fs=sampling_rate, NFFT=256, cmap='viridis')
plt.title('Спектрограмма восстановленного сигнала')
plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц)')

plt.tight_layout()
plt.show()
