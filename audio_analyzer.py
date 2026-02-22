"""
EcoSound AI - Acoustic Ecology Analyzer
Scientific acoustic indices for ecosystem assessment.

Based on:
- ACI: Pieretti et al. 2011
- ADI: Villanueva-Rivera et al. 2011
- BIO: Boelman et al. 2007
- NDSI: Kasten et al. 2012
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from scipy.integrate import trapezoid


class AcousticIndices:
    """Compute scientific acoustic indices for ecosystem assessment."""

    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        self.bio_freq_range    = (2000, 8000)
        self.anthro_freq_range = (200, 2000)

    def compute_spectrogram(self, audio, n_fft=1024, hop_length=512):
        """Compute a power spectrogram and return (freqs, times, Sxx, Sxx_db)."""
        freqs, times, Sxx = signal.spectrogram(
            audio, fs=self.sr, nperseg=n_fft,
            noverlap=n_fft - hop_length, scaling='spectrum'
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        return freqs, times, Sxx, Sxx_db

    def acoustic_complexity_index(self, audio, n_fft=1024, hop_length=512):
        """
        ACI — measures how much sound intensity varies over time.
        Bird song (rapidly changing) → high ACI.
        Engine hum (monotonous)     → low ACI.
        """
        freqs, times, Sxx, _ = self.compute_spectrogram(audio, n_fft, hop_length)

        freq_mask = ((freqs >= self.bio_freq_range[0]) &
                     (freqs <= self.bio_freq_range[1]))
        Sxx_bio = Sxx[freq_mask, :]

        if Sxx_bio.size == 0 or Sxx_bio.shape[1] < 2:
            return 0.0

        aci_per_band = []
        for i in range(Sxx_bio.shape[0]):
            band  = Sxx_bio[i, :]
            diffs = np.sum(np.abs(np.diff(band)))
            total = np.sum(band) + 1e-10
            aci_per_band.append(diffs / total)

        return float(np.mean(aci_per_band))

    def acoustic_diversity_index(self, audio, n_bands=10, n_fft=1024):
        """
        ADI — measures how evenly energy is spread across frequency bands.
        Many species occupying different frequencies → high ADI.
        """
        freqs, _, Sxx, _ = self.compute_spectrogram(audio, n_fft)

        freq_mask       = (freqs >= 200) & (freqs <= 10000)
        freqs_filtered  = freqs[freq_mask]
        Sxx_filtered    = Sxx[freq_mask, :]

        if Sxx_filtered.size == 0:
            return 0.0

        band_edges = np.linspace(
            freqs_filtered[0], freqs_filtered[-1], n_bands + 1
        )
        band_energy = []

        for i in range(n_bands):
            mask = ((freqs_filtered >= band_edges[i]) &
                    (freqs_filtered <  band_edges[i + 1]))
            energy_val = np.sum(Sxx_filtered[mask, :])
            band_energy.append(max(energy_val, 1e-10))

        proportions = np.array(band_energy) / np.sum(band_energy)
        return float(entropy(proportions, base=np.e))

    def bioacoustic_index(self, audio, n_fft=1024):
        """
        BIO — overall biological activity level.
        High sound energy in the 2–8 kHz bird-song band → high BIO.
        """
        freqs, _, Sxx, Sxx_db = self.compute_spectrogram(audio, n_fft)

        freq_mask   = ((freqs >= self.bio_freq_range[0]) &
                       (freqs <= self.bio_freq_range[1]))
        spectrum_db = np.mean(Sxx_db[freq_mask, :], axis=1)

        if len(spectrum_db) == 0:
            return 0.0

        spectrum_normalized = spectrum_db - np.min(spectrum_db)
        return float(trapezoid(spectrum_normalized))

    def ndsi(self, audio, n_fft=1024):
        """
        NDSI — nature vs. human noise ratio.
        +1.0 = pure nature (biophony dominant)
         0.0 = equal mix
        -1.0 = pure machinery (anthrophony dominant)
        """
        freqs, _, Sxx, _ = self.compute_spectrogram(audio, n_fft)

        bio_mask    = ((freqs >= self.bio_freq_range[0]) &
                       (freqs <= self.bio_freq_range[1]))
        bio_energy  = np.sum(Sxx[bio_mask, :])

        anthro_mask   = ((freqs >= self.anthro_freq_range[0]) &
                         (freqs <= self.anthro_freq_range[1]))
        anthro_energy = np.sum(Sxx[anthro_mask, :])

        total = bio_energy + anthro_energy
        if total < 1e-10:
            return 0.0

        return float((bio_energy - anthro_energy) / total)

    def spectral_entropy(self, audio, n_fft=1024):
        """Spectral entropy — unpredictability of energy across frequencies."""
        freqs, _, Sxx, _ = self.compute_spectrogram(audio, n_fft)
        mean_spectrum = np.mean(Sxx, axis=1)
        total         = np.sum(mean_spectrum) + 1e-10
        proportions   = mean_spectrum / total
        return float(entropy(proportions, base=2))

    def temporal_entropy(self, audio, frame_length=1024):
        """Temporal entropy — variability of loudness over time."""
        n_frames = len(audio) // frame_length
        if n_frames < 2:
            return 0.0

        energies = []
        for i in range(n_frames):
            frame = audio[i * frame_length:(i + 1) * frame_length]
            energies.append(np.sum(frame ** 2))

        energies    = np.array(energies)
        total       = np.sum(energies) + 1e-10
        proportions = energies / total
        return float(entropy(proportions, base=2))

    def detect_sound_events(self, audio, threshold_db=-30,
                            min_duration=0.05):
        """Detect individual sound events by energy thresholding."""
        frame_length = int(0.02 * self.sr)
        hop          = frame_length // 2

        energies = []
        for i in range(0, len(audio) - frame_length, hop):
            frame = audio[i:i + frame_length]
            rms   = np.sqrt(np.mean(frame ** 2))
            db    = 20 * np.log10(rms + 1e-10)
            energies.append(db)

        energies = np.array(energies)

        if len(energies) == 0:
            return []

        threshold = np.max(energies) + threshold_db
        active    = energies > threshold
        events    = []
        in_event  = False
        start     = 0

        for i, is_active in enumerate(active):
            if is_active and not in_event:
                start    = i
                in_event = True
            elif not is_active and in_event:
                duration = (i - start) * hop / self.sr
                if duration >= min_duration:
                    events.append({
                        'start_time':  start * hop / self.sr,
                        'end_time':    i * hop / self.sr,
                        'duration':    duration,
                        'peak_energy': float(np.max(energies[start:i]))
                    })
                in_event = False

        return events

    def frequency_band_analysis(self, audio, n_fft=2048):
        """
        Compute energy proportion in each ecological frequency band.

        Band keys are kept in their original form so that visualizer.py
        can map them to short display labels via BAND_LABELS.
        """
        freqs, _, Sxx, _ = self.compute_spectrogram(audio, n_fft)
        total_energy = np.sum(Sxx) + 1e-10

        bands = {
            'geofonia (< 200 Hz)':               (0,    200),
            'ssaki, antropofonia (200-1000 Hz)':  (200,  1000),
            'żaby, duże ptaki (1-2 kHz)':         (1000, 2000),
            'ptaki śpiewające (2-5 kHz)':         (2000, 5000),
            'owady, małe ptaki (5-8 kHz)':        (5000, 8000),
            'nietoperze, owady (> 8 kHz)':        (8000, 11025),
        }

        result = {}
        for name, (low, high) in bands.items():
            mask       = (freqs >= low) & (freqs < high)
            energy_val = np.sum(Sxx[mask, :])

            mean_db = float(np.mean(10 * np.log10(Sxx[mask, :] + 1e-10))) \
                      if np.any(mask) else -100.0

            result[name] = {
                'energy':     float(energy_val),
                'proportion': float(energy_val / total_energy) * 100,
                'mean_db':    mean_db,
            }

        return result