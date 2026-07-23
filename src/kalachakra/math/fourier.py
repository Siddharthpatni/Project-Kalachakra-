"""
Domain 1/26: Mathematical Foundations — Signal Processing & Fourier Analysis

Research References:
    - Cooley & Tukey, "An Algorithm for the Machine Calculation of Complex
      Fourier Series" (1965), Math. Comput. 19(90), 297-301
    - Morlet et al., "Wave propagation and sampling theory" (1982)
    - Huang et al., "The Empirical Mode Decomposition and the Hilbert Spectrum
      for Nonlinear and Non-Stationary Time Series Analysis" (1998),
      Proc. Royal Soc. A 454(1971), 903-995
    - Savitzky & Golay, "Smoothing and Differentiation of Data by Simplified
      Least Squares Procedures" (1964), Analytical Chemistry 36(8), 1627-1639

Planetary motion produces quasi-periodic, non-stationary signals.
This module provides the tools to decompose, filter, and analyze them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

from kalachakra.core.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# Result Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class FFTResult:
    """Result of Fast Fourier Transform.

    Contains frequencies, amplitudes, phases, and power spectrum.
    """

    frequencies: NDArray[np.floating]   # Frequency bins (Hz)
    amplitudes: NDArray[np.floating]    # |F(ω)|
    phases: NDArray[np.floating]        # angle(F(ω))
    power_spectrum: NDArray[np.floating]  # |F(ω)|²
    dominant_frequency: float
    dominant_period: float


@dataclass(frozen=True, slots=True)
class WaveletResult:
    """Result of Continuous Wavelet Transform."""

    coefficients: NDArray[np.complexfloating]  # CWT coefficients
    frequencies: NDArray[np.floating]
    times: NDArray[np.floating]
    power: NDArray[np.floating]  # |coefficients|²


# =============================================================================
# Fourier Transform
# =============================================================================


def fft(
    signal: NDArray[np.floating],
    sample_rate: float = 1.0,
    window: str | None = "hann",
) -> FFTResult:
    """Fast Fourier Transform with windowing.

    Reference: Cooley & Tukey (1965)

    The DFT transforms a sequence of N samples {xₙ} into frequency domain:

        X_k = Σₙ₌₀^{N-1} xₙ · e^{-2πi·kn/N}    for k = 0, 1, ..., N-1

    The FFT computes this in O(N log N) instead of O(N²) using the
    divide-and-conquer approach (radix-2 butterfly operations).

    Inverse DFT:
        xₙ = (1/N) Σₖ₌₀^{N-1} Xₖ · e^{2πi·kn/N}

    Frequency resolution:
        Δf = fs / N

    Nyquist frequency:
        f_max = fs / 2

    Application in Kalachakra:
        Identify dominant periodicities in planetary motion data.
        Detect synodic periods, Dasha cycles, and retrograde frequencies.

    Args:
        signal: Input time-domain signal.
        sample_rate: Sampling rate in Hz (samples per unit time).
        window: Window function to reduce spectral leakage.
                Options: "hann", "hamming", "blackman", "bartlett", None.

    Returns:
        FFTResult with frequencies, amplitudes, phases, power spectrum.
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)

    # Apply window function (reduces spectral leakage)
    if window is not None:
        w = sp_signal.get_window(window, N)
        signal = signal * w

    # Compute FFT (positive frequencies only)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    # Amplitudes and phases
    amplitudes = 2.0 * np.abs(fft_vals) / N
    amplitudes[0] /= 2  # DC component
    phases = np.angle(fft_vals)
    power = amplitudes**2

    # Dominant frequency
    idx_max = np.argmax(amplitudes[1:]) + 1  # Skip DC
    dominant_freq = float(freqs[idx_max])
    dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float("inf")

    log.debug(
        f"FFT: N={N}, dominant_freq={dominant_freq:.4f}, "
        f"period={dominant_period:.4f}"
    )

    return FFTResult(
        frequencies=freqs,
        amplitudes=amplitudes,
        phases=phases,
        power_spectrum=power,
        dominant_frequency=dominant_freq,
        dominant_period=dominant_period,
    )


def inverse_fft(
    fft_result: FFTResult,
    N: int,
) -> NDArray[np.floating]:
    """Inverse FFT: reconstruct time-domain signal from frequency components.

    Formula:
        xₙ = (1/N) Σₖ Xₖ · e^{2πi·kn/N}

    Args:
        fft_result: FFTResult from forward FFT.
        N: Length of original signal.

    Returns:
        Reconstructed time-domain signal.
    """
    # Reconstruct complex spectrum from amplitudes and phases
    spectrum = fft_result.amplitudes * N / 2 * np.exp(1j * fft_result.phases)
    spectrum[0] *= 2  # Undo DC scaling
    return np.fft.irfft(spectrum, n=N)


def stft(
    signal: NDArray[np.floating],
    sample_rate: float = 1.0,
    window_size: int = 256,
    hop_size: int | None = None,
    window: str = "hann",
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.complexfloating]]:
    """Short-Time Fourier Transform.

    Reference: Allen, "Short Term Spectral Analysis, Synthesis, and
               Modification by Discrete Fourier Transform" (1977)

    Formula:
        STFT{x(t)}(τ, ω) = Σₙ x(n) · w(n-τ) · e^{-iωn}

    where w(n) is the window function centered at time τ.

    The STFT provides time-frequency representation at the cost of
    the Heisenberg uncertainty principle:
        Δt · Δf ≥ 1/(4π)

    Application in Kalachakra:
        Track how planetary periodicities evolve over time
        (e.g., retrograde onset detection).

    Args:
        signal: Input signal.
        sample_rate: Sampling rate.
        window_size: Length of each segment.
        hop_size: Number of samples between segments. Default = window_size//4.
        window: Window function name.

    Returns:
        Tuple of (frequencies, times, Zxx spectrogram).
    """
    if hop_size is None:
        hop_size = window_size // 4

    freqs, times, Zxx = sp_signal.stft(
        signal,
        fs=sample_rate,
        window=window,
        nperseg=window_size,
        noverlap=window_size - hop_size,
    )

    return freqs, times, Zxx


# =============================================================================
# Wavelet Transform
# =============================================================================


def cwt(
    signal: NDArray[np.floating],
    sample_rate: float = 1.0,
    freq_range: tuple[float, float] = (0.01, 0.5),
    n_freqs: int = 128,
    wavelet: str = "morl",
) -> WaveletResult:
    """Continuous Wavelet Transform.

    Reference: Morlet et al. (1982), "Wave propagation and sampling theory"
               Torrence & Compo, "A Practical Guide to Wavelet Analysis" (1998)

    The CWT decomposes a signal using scaled/translated wavelets:

        W(a, b) = (1/√a) ∫ x(t) · ψ*((t-b)/a) dt

    where:
        ψ(t) = wavelet mother function
        a = scale parameter (inversely proportional to frequency)
        b = translation parameter (time localization)

    Morlet wavelet:
        ψ(t) = π^(-1/4) · e^{iω₀t} · e^{-t²/2}

    Advantages over STFT:
        - Multi-resolution: high freq → good time resolution
                           low freq → good frequency resolution
        - Natural for non-stationary signals

    Application in Kalachakra:
        Multi-scale analysis of planetary orbital signals.
        Detect transient events (eclipses, conjunctions) at precise times
        while also capturing long-period cycles (Dasha periods, yugas).

    Args:
        signal: Input time-domain signal.
        sample_rate: Sampling rate.
        freq_range: (min_freq, max_freq) in Hz.
        n_freqs: Number of frequency scales.
        wavelet: Wavelet type ("morl" = Morlet, "mexh" = Mexican hat).

    Returns:
        WaveletResult with coefficients, frequencies, times, power.
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    times = np.arange(N) / sample_rate

    # Generate log-spaced frequencies → linear-spaced widths
    freqs = np.logspace(
        np.log10(freq_range[0]),
        np.log10(freq_range[1]),
        n_freqs,
    )

    # Convert frequencies to wavelet widths
    # For Morlet: width ≈ fs / (2π · freq)
    widths = sample_rate / (2 * np.pi * freqs)

    # Compute CWT
    coefficients = sp_signal.cwt(signal, sp_signal.morlet2, widths)

    power = np.abs(coefficients) ** 2

    return WaveletResult(
        coefficients=coefficients,
        frequencies=freqs,
        times=times,
        power=power,
    )


# =============================================================================
# Filtering
# =============================================================================


def butterworth_filter(
    signal: NDArray[np.floating],
    cutoff: float | tuple[float, float],
    sample_rate: float,
    order: int = 4,
    filter_type: str = "low",
) -> NDArray[np.floating]:
    """Butterworth IIR filter.

    Reference: Butterworth, "On the Theory of Filter Amplifiers" (1930)

    The Butterworth filter has maximally flat frequency response in the
    passband. The magnitude response:

        |H(ω)|² = 1 / (1 + (ω/ωc)^{2n})

    where n is the filter order and ωc is the cutoff frequency.

    At the cutoff frequency: |H(ωc)| = 1/√2 ≈ -3dB

    Application in Kalachakra:
        Remove high-frequency noise from planetary position data
        while preserving orbital period signals.

    Args:
        signal: Input signal.
        cutoff: Cutoff frequency (Hz) or (low, high) for bandpass.
        sample_rate: Sampling rate.
        order: Filter order (higher = sharper cutoff, more ringing).
        filter_type: "low", "high", "band", "bandstop".

    Returns:
        Filtered signal.
    """
    nyquist = sample_rate / 2
    if isinstance(cutoff, tuple):
        Wn = (cutoff[0] / nyquist, cutoff[1] / nyquist)
    else:
        Wn = cutoff / nyquist

    b, a = sp_signal.butter(order, Wn, btype=filter_type)
    return sp_signal.filtfilt(b, a, signal)


def savitzky_golay_filter(
    signal: NDArray[np.floating],
    window_length: int = 11,
    poly_order: int = 3,
    deriv: int = 0,
) -> NDArray[np.floating]:
    """Savitzky-Golay smoothing filter.

    Reference: Savitzky & Golay (1964), "Smoothing and Differentiation of
               Data by Simplified Least Squares Procedures"

    Fits a polynomial of order p to a sliding window of width 2m+1,
    then evaluates the polynomial at the center point.

    Equivalent to convolution with specific coefficients derived from
    a least-squares fit. Can also compute derivatives.

    Advantages:
        - Preserves peak height and width better than moving average
        - Can compute smooth derivatives

    Args:
        signal: Input signal.
        window_length: Window length (must be odd, > poly_order).
        poly_order: Polynomial order.
        deriv: Derivative order (0 = smoothing, 1 = first derivative, ...).

    Returns:
        Filtered signal (or derivative if deriv > 0).
    """
    return sp_signal.savgol_filter(signal, window_length, poly_order, deriv=deriv)


# =============================================================================
# Spectral Analysis
# =============================================================================


def periodogram(
    signal: NDArray[np.floating],
    sample_rate: float = 1.0,
    method: str = "welch",
    nperseg: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Power spectral density estimation.

    Methods:
        - "periodogram": |FFT(x)|² / N (high variance)
        - "welch": Averaged modified periodogram (Welch 1967)
          Divides signal into overlapping segments, windows each,
          computes FFT, averages power spectra. Reduces variance.

    Application in Kalachakra:
        Identify which planetary frequencies carry significant power
        vs. noise floor. Used for feature selection.

    Args:
        signal: Input signal.
        sample_rate: Sampling rate.
        method: "periodogram" or "welch".
        nperseg: Segment length for Welch method.

    Returns:
        Tuple of (frequencies, power spectral density).
    """
    if method == "welch":
        return sp_signal.welch(signal, fs=sample_rate, nperseg=nperseg)
    else:
        return sp_signal.periodogram(signal, fs=sample_rate)


def hilbert_transform(
    signal: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Hilbert transform for instantaneous frequency and amplitude.

    Reference: Gabor, "Theory of Communication" (1946)

    The analytic signal:
        z(t) = x(t) + i·H[x(t)]

    where H[x] is the Hilbert transform:
        H[x(t)] = (1/π) · PV ∫ x(τ)/(t-τ) dτ

    From the analytic signal:
        Instantaneous amplitude: A(t) = |z(t)|
        Instantaneous phase: φ(t) = arg(z(t))
        Instantaneous frequency: f(t) = (1/2π) · dφ/dt

    Application in Kalachakra:
        Track the instantaneous speed and amplitude modulation
        of planetary motion, especially during retrograde transitions.

    Args:
        signal: Real-valued input signal.

    Returns:
        Tuple of (amplitude_envelope, instantaneous_phase, instantaneous_freq).
    """
    analytic = sp_signal.hilbert(signal)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi)
    inst_freq = np.append(inst_freq, inst_freq[-1])  # Pad to same length

    return amplitude, phase, inst_freq


def autocorrelation(
    signal: NDArray[np.floating],
    max_lag: int | None = None,
    normalize: bool = True,
) -> NDArray[np.floating]:
    """Compute autocorrelation function.

    Formula:
        R(τ) = E[(X(t) - μ)(X(t+τ) - μ)]

    Normalized:
        ρ(τ) = R(τ) / R(0)   (so ρ(0) = 1)

    Computed efficiently via FFT:
        R = IFFT(|FFT(x)|²)

    Application in Kalachakra:
        Detect periodicities in planetary position time series.
        Peaks in the ACF indicate cyclic patterns.

    Args:
        signal: Input signal.
        max_lag: Maximum lag to compute. None = N//2.
        normalize: If True, normalize so ACF(0) = 1.

    Returns:
        Autocorrelation values for lags 0, 1, ..., max_lag.
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    if max_lag is None:
        max_lag = N // 2

    # Zero-mean
    x = signal - np.mean(signal)

    # FFT-based autocorrelation
    fft_x = np.fft.fft(x, n=2 * N)
    acf = np.real(np.fft.ifft(np.abs(fft_x) ** 2))[:max_lag + 1]

    if normalize and acf[0] > 0:
        acf = acf / acf[0]

    return acf
