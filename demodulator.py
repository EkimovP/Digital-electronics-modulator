import numpy as np
import matplotlib.pyplot as plt


class CostasLoop:
    def __init__(self, initial_phase_estimate=0.0, loop_bandwidth=0.01):
        self.phase_estimate = initial_phase_estimate  # Начальная оценка фазы
        self.loop_bandwidth = loop_bandwidth  # Полоса пропускания петли

    def generate_local_oscillator(self, t, frequency):
        # Генерация локального осциллятора с управляемой частотой
        return np.exp(1j * (2 * np.pi * frequency * t + self.phase_estimate))

    def phase_detector(self, received_signal, local_oscillator):
        # Фазовый детектор (умножение сигналов)
        return received_signal * np.conj(local_oscillator)

    def loop_filter(self, phase_error):
        # Фильтр петли (пропорционально-интегральный фильтр)
        return -self.loop_bandwidth * phase_error

    def update_phase_estimate(self, loop_output):
        # Обновление оценки фазы петли
        self.phase_estimate += loop_output

    def run(self, received_signal, sampling_frequency, frequency):
        # Запуск петли Костаса на входном сигнале
        t = np.arange(len(received_signal)) / sampling_frequency
        output_bits = []

        for i in range(len(received_signal)):
            local_oscillator = self.generate_local_oscillator(t[i], frequency)
            phase_error = self.phase_detector(received_signal[i], local_oscillator)
            loop_output = self.loop_filter(phase_error)
            self.update_phase_estimate(loop_output)

            # Демодуляция битов (по переходу фазы)
            if np.angle(loop_output) >= 0:
                output_bits.append(1)
            else:
                output_bits.append(0)

        return output_bits


# --------------------------------------------------------------------------------------------
# Проверка на синусоиде и меандре

# # Создание объекта петли Костаса
# costas_loop = CostasLoop()

# # Примеры сигналов
# t = np.linspace(0, 1, 1000)
# pure_sine_wave = np.sin(2 * np.pi * 5 * t)  # Чистый синус
# square_wave = np.sign(np.sin(2 * np.pi * 5 * t))  # Меандр

# # Запуск петли Костаса на чистом синусе
# output_bits_sine = costas_loop.run(pure_sine_wave, sampling_frequency=1000, frequency=5)

# # Запуск петли Костаса на меандре
# output_bits_square = costas_loop.run(square_wave, sampling_frequency=1000, frequency=5)

# # Визуализация результатов
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(t, pure_sine_wave, label="Входной сигнал: Чистая синусоидальная волна")
# plt.plot(t, np.array(output_bits_sine) - 1, label="Выходные биты (синуса)")
# plt.title("Выход контура Costas на чистой синусоидальной волне")
# plt.legend()
# print(output_bits_sine)

# plt.subplot(2, 1, 2)
# plt.plot(t, square_wave, label="Входной сигнал: меандр волна")
# plt.plot(t, np.array(output_bits_square) - 1, label="Выходные биты (меандра)")
# plt.title("Выход контура Costas на меандре")
# plt.legend()

# plt.tight_layout()
# plt.show()

# --------------------------------------------------------------------------------------------

# Генерация FM-2 сигнала с шумом и доплеровским сдвигом
np.random.seed(0)

# Параметры сигнала
Fs = 500  # Частота дискретизации
Fc = 10  # Частота несущей
Fdev = 15  # Максимальное девиация частоты
Beta = Fdev / Fc  # Коэффициент модуляции

t = np.linspace(0, 1, Fs)  # Временная ось

# FM-2 модуляция синусоиды
modulated_signal = np.sin(2 * np.pi * Fc * t + Beta * np.sin(2 * np.pi * 2 * t))

# Добавление шума
SNR_dB = 10  # Отношение сигнал/шум в децибелах
noise_power = 10 ** (-SNR_dB / 10)
noise = np.random.normal(0, np.sqrt(noise_power), len(t))
noisy_signal = modulated_signal + noise

# Доплеровский сдвиг
doppler_factor = 0.1
doppler_shift = np.exp(1j * 2 * np.pi * doppler_factor * t)
received_signal = noisy_signal * doppler_shift

# Инициализация и применение петли Костаса
costas_loop = CostasLoop()
output_bits = costas_loop.run(received_signal, Fs, Fc)

# Визуализация результатов
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, modulated_signal, label="Modulated Signal")
plt.title("FM-2 Modulated Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, label="Noisy Signal")
plt.title("Noisy FM-2 Signal with Doppler Shift")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, np.array(output_bits) - 1, label="Output Bits (Recovered)")
plt.title("Recovered Bits using Costas Loop")
plt.xlabel("Time")
plt.ylabel("Bit Value")
plt.yticks([0, 1], ["0", "1"])
plt.legend()
print(output_bits)

plt.tight_layout()
plt.show()
