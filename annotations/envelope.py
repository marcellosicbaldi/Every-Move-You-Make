import numpy as np
import pandas as pd

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Compute high and low envelopes of a signal s
    Parameters
    ----------
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases

    Returns
    -------
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

def return_envelope_diff(acc, envelope = True):
    """
    Detect bursts in acceleration signal

    Parameters
    ----------
    std_acc : pd.Series
        Standard deviation of acceleration signal with a 1 s resolution
    envelope : bool, optional
        If True, detect bursts based on the envelope of the signal
        If False, detect bursts based on the std of the signal

    Returns
    -------
    bursts : pd.Series
        pd.DataFrame with burst start times, end times, and duration
    """

    if envelope:
        lmin, lmax = hl_envelopes_idx(acc.values, dmin=9, dmax=9)
        # adjust shapes
        if len(lmin) > len(lmax):
            lmin = lmin[:-1]
        if len(lmax) > len(lmin):
            lmax = lmax[1:]
        env_diff = pd.Series(acc.values[lmax] - acc.values[lmin], index = acc.index[lmax]) 
        return env_diff