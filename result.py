import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из файлов
with open(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/data.txt", "r"
) as file:
    tuning_sequence_str = file.read().splitlines()[0]

tuning_sequence = np.array([int(bit) for bit in tuning_sequence_str.split()])

bpsk_signal = np.loadtxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/bpsk_signal.txt"
)

# Длина одного бита в отсчётах
bit_length = 250


# Функция для демодуляции BPSK сигнала с использованием схемы Костаса
def demodulate_bpsk(signal, tuning_sequence, bit_length):
    signal_length = len(signal)
    num_bits = signal_length // bit_length

    # Разбиение сигнала на отдельные биты
    bits = np.zeros(num_bits, dtype=int)
    for i in range(num_bits):
        bit_signal = signal[i * bit_length : (i + 1) * bit_length]
        # Комплексный сигнал настройки
        tuning_signal = np.exp(-1j * np.pi * 2 * tuning_sequence.repeat(bit_length))
        # Демодуляция Костаса
        I = np.real(bit_signal * np.conj(tuning_signal))
        Q = np.imag(bit_signal * np.conj(tuning_signal))
        bits[i] = int(np.mean(I) < 0)

    return bits


# Демодуляция BPSK сигнала
recovered_bits = demodulate_bpsk(bpsk_signal[:, 1], tuning_sequence, bit_length)

# Визуализация
plt.figure(figsize=(10, 5))

# Исходный сигнал
plt.subplot(2, 1, 1)
plt.plot(bpsk_signal[:, 0], bpsk_signal[:, 1])
plt.title("Исходный BPSK сигнал")
plt.xlabel("Время")
plt.ylabel("Амплитуда")

# Восстановленная битовая последовательность
plt.subplot(2, 1, 2)
plt.stem(recovered_bits, use_line_collection=True)
plt.title("Восстановленная битовая последовательность")
plt.xlabel("Отсчёт")
plt.ylabel("Значение бита")

plt.tight_layout()
plt.show()

print(recovered_bits)
print(len(recovered_bits))
