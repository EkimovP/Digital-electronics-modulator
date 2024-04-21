import numpy as np
import matplotlib.pyplot as plt


class Modulator:
    def __init__(self, signal, carrier_freq, sampling_rate):
        self.signal = signal
        self.carrier_freq = carrier_freq
        self.sampling_rate = sampling_rate

    def recover_carrier(self):
        # Восстановление несущей частоты (Carrier Recovery)
        t = np.arange(len(self.signal)) / self.sampling_rate
        carrier = np.exp(1j * 2 * np.pi * self.carrier_freq * t)
        return self.signal * carrier

    def apply_low_pass_filter(self, signal):
        # Применение фильтра нижних частот (Low Pass Filter)
        # Простой фильтр нижних частот (скользящее среднее)
        cutoff_freq = 0.2 * self.carrier_freq
        num_taps = int(
            0.1 * self.sampling_rate / cutoff_freq
        )  # Длина окна скользящего среднего
        filter_window = np.ones(num_taps) / num_taps
        filtered_signal = np.convolve(signal, filter_window, mode="same")
        return filtered_signal

    def thresholding(self, signal):
        # Пороговый элемент (Thresholding)
        thresholded_signal = np.where(np.real(signal) > 0, 1, 0)
        return thresholded_signal

    def recover_clock(self, signal):
        # Восстановление тактовой частоты (Clock Recovery)
        # Извлечение тактового сигнала методом максимального правдоподобия (MLSE)
        phase_diff = np.angle(signal[1:] * np.conj(signal[:-1]))
        phase_diff = np.unwrap(
            phase_diff
        )  # Развертка фазы для устранения перепадов на 2*pi

        # Восстановление тактовой частоты путем интерполяции фазовых различий
        interpolated_clock = np.interp(
            np.arange(len(self.signal)), np.arange(len(phase_diff)), phase_diff
        )
        clock_signal = np.exp(-1j * interpolated_clock)

        return clock_signal

    def modulate(self):
        # Основной процесс модуляции
        carrier_recovered = self.recover_carrier()
        filtered_signal = self.apply_low_pass_filter(carrier_recovered)
        thresholded_signal = self.thresholding(filtered_signal)
        clock_signal = self.recover_clock(filtered_signal)
        binary_data = self.thresholding(
            thresholded_signal * clock_signal
        )  # Применение тактового сигнала к бинарным данным
        return binary_data


# Пример использования модулятора:
if __name__ == "__main__":
    # Параметры сигнала
    duration = 1  # Длительность сигнала в секундах
    fs = 500  # Частота дискретизации в Гц
    f_carrier = 20  # Частота несущей синусоиды в Гц
    bit_rate = 25  # Скорость передачи битов в битах в секунду (число бит в периоде)

    # Генерируем последовательность битов (для примера)
    bits = np.random.randint(0, 2, int(bit_rate * duration))

    # Генерируем временную ось
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Генерируем несущую синусоиду
    carrier_wave = np.sin(2 * np.pi * f_carrier * t)

    # Генерируем модулированный сигнал
    modulated_signal = np.zeros_like(carrier_wave)

    bit_period_samples = int(fs / bit_rate)  # Число сэмплов в одном периоде бита

    for i, bit in enumerate(bits):
        phase_shift = (
            np.pi if bit == 1 else 0
        )  # Фазовый сдвиг на 180 градусов при бите 1
        modulated_signal[i * bit_period_samples : (i + 1) * bit_period_samples] = (
            np.sin(
                2
                * np.pi
                * f_carrier
                * t[i * bit_period_samples : (i + 1) * bit_period_samples]
                + phase_shift
            )
        )

    # Добавляем шум
    SNR_dB = 10  # Отношение сигнал/шум в децибелах
    signal_power = np.sum(modulated_signal**2) / len(modulated_signal)
    noise_power = signal_power / (10 ** (SNR_dB / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), modulated_signal.shape)
    noisy_signal = modulated_signal + noise

    # Моделируем эффект доплеровского сдвига
    doppler_factor = 2.0  # Коэффициент доплеровского сдвига
    doppler_shift = np.cos(
        2 * np.pi * doppler_factor * t
    )  # Модулирующая функция для доплеровского сдвига
    doppler_signal = noisy_signal * doppler_shift

    # # Параметры сигнала и модулятора
    # carrier_freq = 10  # несущая частота (Гц)
    # sampling_rate = 500  # частота дискретизации (Гц)
    # duration = 1.0  # длительность сигнала в секундах

    # # Создание FM-2 модулированной синусоиды с шумом и эффектом доплера
    # np.random.seed(0)
    # t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

    # fm_deviation = 5  # размах частотной модуляции
    # modulation_freq = 10  # частота модуляции
    # carrier_signal = np.sin(
    #     2 * np.pi * carrier_freq * t
    #     + fm_deviation * np.sin(2 * np.pi * modulation_freq * t)
    # )

    # # Добавление шума
    # noise_power = 0.01 * np.var(
    #     carrier_signal
    # )  # мощность шума (10% от дисперсии сигнала)
    # noisy_signal = carrier_signal + np.random.normal(
    #     scale=np.sqrt(noise_power), size=len(carrier_signal)
    # )

    noisy_signal = doppler_signal

    # Создание экземпляра модулятора
    modulator = Modulator(noisy_signal, f_carrier, fs)

    # Извлечение бинарных данных из зашумленного сигнала
    binary_data_output = modulator.modulate()

    # Построение графиков
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, modulated_signal, label="Синусоида с BPSK")
    plt.title("Сигнал исходный без шума и доплеровского сдвига")
    plt.xlabel("Время")
    plt.ylabel("Амплитуда")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_signal, label="Сигнал")
    plt.title("Синусоида с шумом и доплеровским сдвигом")
    plt.xlabel("Время")
    plt.ylabel("Амплитуда")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, binary_data_output.real, label="Биты")
    plt.title("Последовательность битов")
    plt.xlabel("Время")
    plt.ylabel("Биты")
    plt.ylim([0, 1.1])
    plt.legend()
    print(binary_data_output.real)

    plt.tight_layout()
    plt.show()
