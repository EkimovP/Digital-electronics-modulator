import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Загрузка исходных данных из файла (сигнала)
bpsk_signal = np.loadtxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/filtered_bpsk_signal_var2.txt"
)
# Длина одного бита в отсчётах
bit_length = 250


# Функция для создания фильтра нижних частот
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


# Функция для создания полосового фильтра
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


# Функция для применения фильтра к сигналу
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Реализация схемы Костаса для демодуляции исходного сигнала
def costas_loop(signal, Kp, Ki):
    N = len(signal)
    phase_error = np.zeros(N)
    integrator = np.zeros(N)
    phase_est = np.zeros(N)
    freq_est = np.zeros(N)
    phase = 0
    freq = 0

    for i in range(1, N):
        integrator[i] = integrator[i - 1] + phase_error[i - 1]
        phase_est[i] = (
            phase_est[i - 1] + freq + Kp * phase_error[i] + Ki * integrator[i]
        )
        phase = np.mod(phase_est[i], 2 * np.pi)
        freq_est[i] = freq
        I = signal[i] * np.cos(phase)
        Q = signal[i] * np.sin(phase)
        phase_error[i] = I * Q
        freq += Kp * phase_error[i]

    return phase_est


# Демодуляция
def demodulate_bpsk(signal, bit_length, lowcut, highcut, fs):
    signal_length = len(signal)
    num_bits = signal_length // bit_length
    bits = np.zeros(num_bits, dtype=int)
    Kp = 0.1
    Ki = 0.01
    phase_est = costas_loop(signal, Kp, Ki)

    for i in range(num_bits):
        bit_signal = signal[i * bit_length : (i + 1) * bit_length]
        I = bit_signal * np.cos(phase_est[i * bit_length : (i + 1) * bit_length])
        Q = bit_signal * np.sin(phase_est[i * bit_length : (i + 1) * bit_length])
        I_filtered = bandpass_filter(I, lowcut, highcut, fs)
        bits[i] = int(np.mean(I_filtered) < 0)

    return bits


# ПСП
psp = np.array([0, 0, 1, 0, 1, 1, 1])

# Демодуляция BPSK сигнала с учетом полосового фильтра
lowcut = 0.1  # нижняя частота среза
highcut = 0.3  # верхняя частота среза
fs = 1  # частота дискретизации (может потребоваться настроить)

recovered_bits = demodulate_bpsk(bpsk_signal[:, 1], bit_length, lowcut, highcut, fs)

# Загрузка эталонной последовательности
data_package = np.loadtxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/data_package_var2.txt",
    dtype=int,
)

# Сравнение демодулированной последовательности с эталонной
if len(recovered_bits) != len(data_package):
    print("Разное количество битов! Проверка не будет выполнена!")
else:
    correct_bits = np.sum(recovered_bits == data_package)
    total_bits = len(data_package)
    accuracy = correct_bits / total_bits * 100
    print(f"Точность: {accuracy:.2f}%")

# Корреляция с ПСП
psp_length = len(psp)
correlation = np.correlate(recovered_bits, psp, mode="valid")
start_index = np.argmax(correlation)

# Извлечение информационных бит
informational_bits = recovered_bits[
    start_index + psp_length : start_index + psp_length + len(data_package) - psp_length
]

# Вывод восстановленной битовой последовательности и её длины
print(recovered_bits)
print(len(recovered_bits))

# Вывод информационных битов и их длины
print("Информационные биты:", informational_bits)
print("Длина информационных битов:", len(informational_bits))

# Визуализация
plt.figure(figsize=(10, 7))

# Исходный сигнал
plt.subplot(3, 1, 1)
plt.plot(bpsk_signal[:, 0], bpsk_signal[:, 1])
plt.title("Исходный BPSK сигнал")
plt.xlabel("Время")
plt.ylabel("Амплитуда")

# Восстановленная битовая последовательность
plt.subplot(3, 1, 2)
plt.stem(recovered_bits, use_line_collection=True)
plt.title("Восстановленная битовая последовательность")
plt.xlabel("Отсчёт")
plt.ylabel("Значение бита")

# Информационные биты
plt.subplot(3, 1, 3)
plt.stem(informational_bits, use_line_collection=True)
plt.title("Информационные биты")
plt.xlabel("Отсчёт")
plt.ylabel("Значение бита")

plt.tight_layout()
plt.show()
