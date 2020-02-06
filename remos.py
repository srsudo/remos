_authors__ ="Angel Bueno Rodriguez, Alejandro Diaz Moreno and Silvio De Angelis"
__email__ = "angelbueno@ugr.es"

"""REMOS algorithm main script. 

This script allows the user to use the REMOS segmentation algorithm as described in:

Recursive Entropy Method of Segmentation for Seismic Signals. A. Bueno1, A. Diaz-Moreno, S. De Angelis, 
C. Benitez and J.M.Ibanez. Seismological Research Letters (SRL). 

It is assumed that the seismic data streams has been previously processed and stored in an correct format 
(e.g: miniseed). In practice, if the data can be stored in NumPy format, it can be processed in Python. 

This script requires that `obspy`, `scikits`, `scipy` to be installed within the Python environment you are running 
this algorithm in.

This file can also be imported as a module and contains the following functions:

    * clean_data - returns the segmented candidates that are above or below a threshold.
    * short_term_energy - returns the short term energy of a signal.
    * energy_per_frame - returns an array containing the energy of the framed signal. 
    * plot_signals - Plots a signal, which can be saved or not. 
    * do_sta_lta - Perform a STA/LTA step to obtain the activation time. 
    * run_remos - Runs the REMOS segmentation algorithm. returns the segmented candidates. 
    
"""

import copy
import numpy as np
import obspy
from scikits.talkbox import segment_axis  # it will be deprecated in a future.
import scipy
from scipy.stats import entropy as sci_entropy
import matplotlib.pyplot as plt
import matplotlib
from obspy.signal.trigger import trigger_onset, recursive_sta_lta


def clean_data(candidate_segmented, fm=100.0, snr_thr=30.0, min_duration=10.0):
    """

    Parameters
    ----------
    candidate_segmented: Numpy Array
        A numpy array containing the segmented candidates from REMOS.
    fm: float
        The sampling frequency of the candidates.
    snr_thr: float
        The minimum SNR requirement for the candidate
    min_duration: float
        Duration (in seconds) to be considered.

    Returns
    -------
    list
        A list containing just the selected, final candidates.
    """

    new_candidate = copy.copy(candidate_segmented)
    not_wanted = set()

    for k, m in enumerate(candidate_segmented):
        ff = m[0]
        ff = ff - np.mean(ff)
        upper_root = np.sqrt(1 / float(len(ff[0:2000]) * np.sum(np.power(ff[0:2000], 2))))

        noise_root = np.sqrt(1 / float(len(ff[-int(2.0) * int(fm):]) * np.sum(np.power(ff[-int(2.0) * int(fm):], 2))))

        snr = 10 * np.log(upper_root / noise_root)
        samples = len(ff)

        if (len(ff) / float(fm)) <= min_duration:
            not_wanted.add(k)
        elif np.abs(snr) <= snr_thr:
            not_wanted.add(k)
        else:
            pass

    # we need to iterate over new_candidate to avoid mistakes
    return [m for e, m in enumerate(new_candidate) if e not in not_wanted]


def short_term_energy(chunk):
    """Function to compute the short term energy of a signal as the sum of their squared samples.
    Parameters
    ----------
    chunk: Numpy array
        Signal we would like to compute the signal from
    Returns
    -------
    float
        Containing the short term energy of the signal.
    """
    return np.sum((np.abs(chunk) ** 2) / chunk.shape[0])





def energy_per_frame(windows):
    """It computes the energy per-frame for a given umber of frames.

    Parameters
    ----------
    windows: list
        Containing N number of windows from the seismic signal
    Returns
    -------
    Numpy Array
        Numpy matrix, size N x energy, with N the number of windows, energy their associate energy.
    """
    out = []
    for row in windows:
        out.append(short_term_energy(row))
    return np.hstack(np.asarray(out))


def compute_fft(signal, fm):
    """Function to compute the FFT.
    Parameters
    ----------
    signal: Numpy Array.
        The signal we want to compute the fft from.
    fm: float
        the sampling frequency
    Returns
    -------
    Y: Numpy Array
        The normalized fft
    frq: Numpy Array
        The range of frequencies
    """
    n = len(signal)  # length of the signal
    k = np.arange(n)
    T = n / fm
    frq = k / T  # two sides frequency range
    frq = frq[range(n / 2)]  # one side frequency range
    Y = np.fft.fft(signal) / n  # fft computing and normalization
    Y = Y[range(n / 2)]
    return Y, frq


def plot_signals(signal, label, save=None):
    """Function to plot the a seismic singal as a nu,py array.
    Parameters
    ----------
    signal Numpy Array
         Contains the signals we want to plot
    label: str
         A string containing the label or the signal (or the graph title).
    save: str, optional
        A string containing the picture name we want to save
    -------
    """

    plt.figure(figsize=(10, 8))
    plt.title(label)
    plt.plot(signal)

    if save is not None:
        plt.savefig(save + ".png")
    else:
        plt.show()

    plt.close()


def do_sta_lta(reading_folder, nsta, nlta, trig_on, trig_of, filtered=True, bandpass=None):
    """Function to perform STA/LTA on a given trace and recover the data. It assumess a bandpass filter.
    Other functions/parameters to process the data are encouraged.
    Parameters
    ----------
    reading_folder: basestring
         A string indicating the location amd the name of out data (e.g; data/MV.HHZ.1997.01)
    nsta: float
        Length of short time average window in samples
    nlta: float
        Length of long time average window in samples
    trig_on:
        Frequency to trig on the STA/LTA.
    trig_of:
        Frequency to trig off the STA/LTA.
    filtered: bool, optional
        If we want to apply a bandpass filter to the trace or not.
    bandpass: list, optional
        A list containing the flow and fhigh in which we want to bandpass the signal

    Returns
    -------
    original: Stream Obspy Object.
        The copy of the original data,
    st: Stream Obspy Object.
        The processed, filtered stream.
    data: Numpy Array
        The processed data in array format.
    on_of: Numpy Array
        The number of detections and on/of onsets, size N x 2, with N the number of detections.
    """

    st = obspy.read(reading_folder)  # 1 trace within 1 Stream
    original = st.copy()  # we copy the stream in memory to avoid potential NUmnpy missplacements.
    fm = float(st[0].stats.sampling_rate)

    if filtered:
        st = st.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1])

    data = st[0]
    data = np.asarray(data)

    cft = recursive_sta_lta(data, int(nsta * fm), int(nlta * fm))
    on_of = trigger_onset(cft, trig_on, trig_of)

    return original, st, data, cft, on_of


def run_remos(stream, data, on_of, delay_in, durations_window, epsilon=2.5, plot=False, cut="original"):
    """ Function that executes the main segmentation. Additional pre-processing steps might be required. Please, refer
    to the main manuscript, or github.com/srsudo/remos for additional examples.
    Parameters
    ----------
    stream: Stream Obspy
        The original Stream Obspy object
    data: Numpy Array
        The PROCESSED data from the STA/LTA method
    on_of:
        The numpy matrix, size nsamples x 2, containing the timing
    delay_in: float
        The offset defined to cut from the estimated number of windows
    durations_window: list
        An array containing [W_s, W_d] the window search duration and the minimum window.
    epsilon: float, optional
        The threshold value for the entropy
    plot: bool, optional
        True if we want to plot eacf ot the segmented signals. Be vareful for long streams (>25 min)
    cut:string, optional
        "original" to cut from the main trace, or "processed" to cut from the STA/LTA filtered trace.
    Returns
    X: list
        A list containing the [signal, FI_ratio, start, end]
    -------
    """

    # we make a copy in memory of the original array
    array_original = stream[0].copy()

    # mean removal, high_pass filtering of earth noise background
    array_original = array_original.detrend(type='demean')
    array_original.data = obspy.signal.filter.highpass(array_original.data, 0.5, df=array_original.stats.sampling_rate)

    # plot_signals(array_final[0:10000], "DATA-REMOS-final")

    fm = float(array_original.stats.sampling_rate)
    X = []

    window_size = durations_window[0]
    search_window = durations_window[1]

    data = data - np.mean(data)
    processed_data = data.copy()

    # use the percentile to reduce background
    umbral = np.percentile(data, 80)
    data = (data / float(umbral) * (data > umbral)) + (0 * (data <= umbral))

    for m, k in enumerate(on_of):

        # c = c + 1
        start = int(k[0])
        end = int(k[1])

        x_0 = int(start - delay_in * fm)
        x_1 = np.abs(x_0 - int(start - delay_in * fm + end + search_window * fm))

        # x_1 = np.abs(x_0-int(start+search_window*fm))

        selected_candidate = np.asarray(data[x_0:x_1])
        ventanas = segment_axis(selected_candidate.flatten(), int(window_size * fm), overlap=0)
        energy_ventanas = energy_per_frame(ventanas)
        total_energy = np.sum(energy_ventanas)
        loq = energy_ventanas / float(total_energy)

        if sci_entropy(loq) < epsilon:
            cut_me = int(np.argmin(loq) * window_size * fm + delay_in * fm)
            potential_candidate = array_original[x_0:cut_me + x_0]
            duration_candidate = potential_candidate.shape[0] / float(fm)

            if duration_candidate < 5.0:
                # By doing this, we erase those windows with small durations
                pass

            else:

                potential_candidate = potential_candidate - np.mean(potential_candidate)
                ventanas_ref = segment_axis(potential_candidate.flatten(), int(5.0 * fm), overlap=0)

                try:
                    dsai = int((ventanas_ref.shape[0] / 2.0))
                except:
                    dsai = 0

                try:
                    down_windows = energy_per_frame(ventanas_ref[0:dsai])
                    upper_windows = energy_per_frame(ventanas_ref[dsai:dsai + dsai])
                    ratio = np.round(np.sum(np.asarray(upper_windows)) / np.sum(np.asarray(down_windows)), 2)
                except:
                    ratio = np.inf
                    pass

                if ratio < 0.15:
                    # In this case, long-segmentation, re-cut.
                    try:
                        ind = np.sort(np.argpartition(upper_windows, 2)[:2])[0]
                    except:
                        # print "on exception"
                        ind = upper_windows.shape[0]

                    min_duration = (down_windows.shape[0] + ind) * 5.0 * fm
                    cut_me = int(min_duration)

            if cut == "original":
                selected_candidate = array_original.data[x_0:cut_me + x_0]
                X.append([selected_candidate, ratio, x_0, cut_me + x_0])
            else:
                selected_candidate = processed_data[x_0:cut_me + x_0]
                X.append([selected_candidate, ratio, x_0, cut_me + x_0])

            # lets plot
            if plot:
                plt.figure()
                plot_signals(selected_candidate, label="SEGMENTED")
        else:
            pass

    return X


def visualize_segmentation(data, positions):
    """Function to plot the data and visualize the result of segmentation, over the real trace.
    Parameters
    ----------
    data: Numpy array
        The seismic signal as a Numpy array we would like to visualize.
    positions:
        A list containing the on_off triggering data.
    Returns
    -------
    """
    plt.figure(figsize=(20, 8))
    ax = plt.subplot()
    segmented_on = positions[:, 0]
    segmented_off = positions[:, 1]
    plt.title("VISUALIZATION OF SEGMENTATION RESULTS")
    plt.plot(data)
    ymin, ymax = ax.get_ylim()
    plt.vlines(segmented_on, ymin, ymax, color='green', linewidth=2, linestyle='solid')
    plt.vlines(segmented_off, ymin, ymax, color='magenta', linewidth=2, linestyle='dashed')
    plt.savefig('sta_lta_continua.eps', format='eps', dpi=300)
    plt.xlim((0, data.shape[0]))  # set the xlim to left, right
    plt.show()