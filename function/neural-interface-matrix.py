import numpy as np
from scipy.signal import butter, lfilter

def create_nim(brain_activity, interface_weights):
    """
    Creates a Neural Interface Matrix (NIM) to interface directly with the human brain.
    
    Parameters:
    brain_activity (np.array): A 1D array representing the brain activity.
    interface_weights (np.array): A 1D array representing the weights of the interface.
    
    Returns:
    np.array: A 2D array representing the NIM.
    """
    
    # Apply a bandpass filter to the brain activity data
    brain_activity = butter_bandpass_filter(brain_activity, lowcut=0.5, highcut=50.0, fs=1000)
    
    # Create the NIM by multiplying the brain activity data with the interface weights
    nim = np.outer(brain_activity, interface_weights)
    
    return nim

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a bandpass filter to the input data.
    
    Parameters:
    data (np.array): A 1D array representing the input data.
    lowcut (float): The low cutoff frequency of the filter.
    highcut (float): The high cutoff frequency of the filter.
    fs (float): The sampling frequency of the data.
    order (int): The order of the filter. Default is 5.
    
    Returns:
    np.array: The filtered data.
    """
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    
    return y
