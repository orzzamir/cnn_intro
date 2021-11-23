import random
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle


def get_dataset():
    with open('data\CNN_data.pkl', 'rb') as f:
        x_train, y_train = pickle.load(f)
    return x_train, y_train


def get_set_size():
    return 10000


def create_gaussian(time, N, miu, sig, amp):
    return [amp * np.exp(-(t - miu) ** 2 / (2 * sig)) for t in time]


def create_pulses(time, N, number_of_pulses, width):
    deltas = np.zeros(N)
    space = N / number_of_pulses
    for i in np.arange(number_of_pulses):
        deltas[int(i * space + 10) % N] = 1
    pulse = create_gaussian(time, N, 0.5, width, 1)
    return ifft(fft(pulse) * fft(deltas))


def create_ft(time, damping, oscillation):
    return [np.exp(-damping * t) * np.cos(oscillation * t) for t in time]


def generate_set(M):
    # Signal Parameters
    N = 2 ** 10
    width = 0.001 / N
    time = np.linspace(0, 1, N)

    # TF Parameters
    damping = 100
    oscillation = 450

    # Noise Parameters
    noise_factor = 1
    tf_factor = 0.1
    pulses_range = 20

    # Construct Signal Parameters
    noise = np.random.normal(0, 1, N) * noise_factor
    sample_damping = damping * (1 + random.random() * 2 * tf_factor - tf_factor)
    sample_oscillation = oscillation * (1 + random.random() * 2 * tf_factor - tf_factor)
    number_of_pulses = 15 + random.randrange(pulses_range)

    # Construct Signal
    signal = np.zeros([M, N])
    tweaks = np.random.normal(0.1, 1, M)

    if min(tweaks) < 0:
        tweaks = (tweaks - min(tweaks) + 1)
    tweaks = tweaks / max(tweaks)

    for m in np.arange(M):
        signal[m][:] = generate_sample(time, N, width, tweaks[m],
                                       sample_damping, sample_oscillation, number_of_pulses, noise)

    return signal, tweaks


def generate_sample(time, N, width, tweak, damping, oscillation, number_of_pulses, noise):
    warnings.filterwarnings('ignore')

    # Generate Transfer Function
    sample_tf = create_ft(time, damping, oscillation)

    # Generate Signal
    pulses = create_pulses(time, N, number_of_pulses, width)
    signal = ifft(fft(sample_tf) *
                  fft(np.add(pulses, create_gaussian(time, N, 0.5+0.45*(random.random()*2-1), tweak * width * 5, tweak * 5))))
    
    '''
    fig, ax = plt.subplots(3, sharey='row')
    ax[0].plot(np.add(pulses, create_gaussian(time, N, 0.5, tweak * width * 5, tweak * 5)))
    ax[0].set_title("Signal Before Transfer Function")
    ax[1].plot(sample_tf)
    ax[1].set_title("The Transfer Function")
    ax[2].plot(signal)
    ax[2].set_title("Final Signal")
    fig.tight_layout()
    plt.show()
    '''

    # Add Noise
    noised = np.add(signal, noise)
      
    return noised


def generate():
    warnings.filterwarnings('ignore')

    # Signal Parameters
    N = 2 ** 10
    width = 0.001 / N
    time = np.linspace(0, 1, N)

    # TF Parameters
    damping = 100
    oscillation = 450
    tf_factor = 0.1
    noise_factor = 0.1
    pulses_range = 20

    # Construct Signal Parameters
    noise = np.random.normal(0, 1, N) * noise_factor
    sample_damping = damping * (1 + random.random() * 2 * tf_factor - tf_factor)
    sample_oscillation = oscillation * (1 + random.random() * 2 * tf_factor - tf_factor)
    number_of_pulses = 15 + random.randrange(pulses_range)

    # Creating Signal
    signal = generate_sample(time, N, width, 0.7, sample_damping, sample_oscillation, number_of_pulses, noise)
    return signal


def generate_feature_test_set(M):
    # Signal Parameters
    N = 2 ** 10
    width = 0.001 / N
    time = np.linspace(0, 1, N)

    # TF Parameters
    damping = 100
    oscillation = 450

    # Noise Parameters
    noise_factor = 1
    tf_factor = 0.1
    pulses_range = 20

    # Construct Signal Parameters
    noise = np.random.normal(0, 1, N) * noise_factor
    sample_damping = damping * (1 + random.random() * 2 * tf_factor - tf_factor)
    sample_oscillation = oscillation * (1 + random.random() * 2 * tf_factor - tf_factor)
    number_of_pulses = 15 + random.randrange(pulses_range)

    # Construct Signal
    signal = np.zeros([M, N])
    tweaks = np.linspace(0.1, 1, M)

    for m in np.arange(M):
        signal[m][:] = generate_sample(time, N, width, tweaks[m],
                                       sample_damping, sample_oscillation, number_of_pulses, noise)

    return signal, tweaks


def test_feature(feature_func, feature_name="Feature"):
    print("Testing " + feature_name + " feature...")
    number_of_tests = 20
    number_of_defects = 30
    for i in np.arange(number_of_tests):
        signal, defect_sizes = generate_feature_test_set(number_of_defects)
        feature = []
        for m in np.arange(number_of_defects):
            feature = feature + [feature_func(signal[m][:])]
        plt.plot(feature, defect_sizes, 'o')
    plt.title("Defect Size vs " + feature_name)
    plt.xlabel(feature_name)
    plt.ylabel("Defect Size")
    plt.show()
