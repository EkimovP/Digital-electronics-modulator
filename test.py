import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Загрузка данных из файлов
with open(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/data.txt", "r"
) as file:
    tuning_sequence_str = file.read().splitlines()[0]

tuning_sequence = np.array([int(bit) for bit in tuning_sequence_str.split()])

bpsk_signal = np.loadtxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/filtered_bpsk_signal_var2.txt"
)

# Длина одного бита в отсчётах
bit_length = 250


# Функция для полосовой фильтрации
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Функция для демодуляции BPSK сигнала с использованием схемы Костаса
def costas_loop(signal, Kp, Ki, sampling_freq):
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


def demodulate_bpsk(signal, bit_length, fs):
    signal_length = len(signal)
    num_bits = signal_length // bit_length

    # Разбиение сигнала на отдельные биты
    bits = np.zeros(num_bits, dtype=int)
    Kp = 0.1
    Ki = 0.01
    phase_est = costas_loop(signal, Kp, Ki, fs)

    for i in range(num_bits):
        bit_signal = signal[i * bit_length : (i + 1) * bit_length]
        # Демодуляция Костаса
        I = bit_signal * np.cos(phase_est[i * bit_length : (i + 1) * bit_length])
        Q = bit_signal * np.sin(phase_est[i * bit_length : (i + 1) * bit_length])
        bits[i] = int(np.mean(I) < 0)

    # Инвертирование битовой последовательности
    bits = 1 - bits

    return bits


# Применение полосового фильтра к входному сигналу
fs = 50
lowcut = 0.4
highcut = 1.6
filtered_signal = bandpass_filter(bpsk_signal[:, 1], lowcut, highcut, fs, order=5)

# Демодуляция BPSK сигнала после полосового фильтра
recovered_bits = demodulate_bpsk(filtered_signal, bit_length, fs)

data_package = np.loadtxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/data_package_var2.txt",
    dtype=int,
)

# Сравнение демодулированной последовательности с эталонной
correct_bits = np.sum(recovered_bits == data_package)
total_bits = len(data_package)
accuracy = correct_bits / total_bits * 100

print(f"Точность: {accuracy:.2f}%")

print(recovered_bits)
print(len(recovered_bits))

# Визуализация
plt.figure(figsize=(10, 7))

# Исходный сигнал
plt.subplot(3, 1, 1)
plt.plot(bpsk_signal[:, 0], bpsk_signal[:, 1])
plt.title("Исходный BPSK сигнал")
plt.xlabel("Время")
plt.ylabel("Амплитуда")

# Отфильтрованный сигнал
plt.subplot(3, 1, 2)
plt.plot(bpsk_signal[:, 0], filtered_signal)
plt.title("Отфильтрованный сигнал")
plt.xlabel("Время")
plt.ylabel("Амплитуда")

# Восстановленная битовая последовательность
plt.subplot(3, 1, 3)
plt.stem(recovered_bits, use_line_collection=True)
plt.title("Восстановленная битовая последовательность")
plt.xlabel("Отсчёт")
plt.ylabel("Значение бита")

plt.tight_layout()
plt.show()
