"""
EcoSound AI - Bird species identification.
Uses BirdNET-Analyzer (Cornell Lab of Ornithology).

INSTALLATION (optional):
    pip install birdnet

This module is OPTIONAL - the rest of the program works without it.
"""

import numpy as np


def is_birdnet_available():
    """Return True if the BirdNET package is installed."""
    try:
        from birdnet import predict
        return True
    except ImportError:
        return False


def identify_birds(audio, sr, min_confidence=0.25, lat=52.0, lon=21.0):
    """
    Identify bird species present in an audio recording.

    Args:
        audio:          numpy array of audio samples
        sr:             sample rate (Hz)
        min_confidence: minimum detection confidence threshold (0–1)
        lat:            recording latitude  (default: Warsaw)
        lon:            recording longitude (default: Warsaw)

    Returns:
        List of dicts with identified species sorted by confidence.
    """

    if not is_birdnet_available():
        print("  ⚠️  BirdNET is not installed")
        print("  💡 Install it with: pip install birdnet")
        print("  💡 The program works without it - drop the --birds flag")
        return []

    try:
        from birdnet import predict
        import tempfile
        import soundfile as sf
        import os

        print("  🐦 BirdNET - identifying bird species...")
        print(f"     Location: {lat}°N, {lon}°E")

        # BirdNET requires a .wav file on disk
        with tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False
        ) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, sr)

        predictions = predict.predict(
            tmp_path,
            lat=lat,
            lon=lon,
            min_conf=min_confidence
        )

        # Clean up temporary file
        os.unlink(tmp_path)

        # Aggregate detections per species
        species_found = {}

        for timestamp, species_list in predictions.items():
            for species_name, confidence in species_list.items():
                if species_name not in species_found:
                    species_found[species_name] = {
                        'scientific_name': species_name,
                        'common_name':     species_name,
                        'confidence':      confidence,
                        'detections':      1,
                        'first_heard':     timestamp,
                    }
                else:
                    species_found[species_name]['detections'] += 1
                    species_found[species_name]['confidence'] = max(
                        species_found[species_name]['confidence'],
                        confidence
                    )

        # Sort by confidence descending
        results = sorted(
            species_found.values(),
            key=lambda x: x['confidence'],
            reverse=True
        )

        if results:
            print(f"     ✅ Identified {len(results)} species:")
            for bird in results[:10]:
                print(
                    f"        🐦 {bird['common_name']} "
                    f"({bird['confidence']:.0%} confidence, "
                    f"{bird['detections']}x)"
                )
        else:
            print("     ℹ️  No bird species identified")
            print("        (recording may be too short or contain no birds)")

        return results

    except Exception as e:
        print(f"  ❌ BirdNET error: {e}")
        return []


def identify_birds_simple(filepath, min_confidence=0.25,
                           lat=52.0, lon=21.0):
    """
    Simplified wrapper - pass a file path instead of an audio array.
    """
    if not is_birdnet_available():
        print("  ⚠️  BirdNET not installed: pip install birdnet")
        return []

    try:
        import soundfile as sf
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return identify_birds(audio, sr, min_confidence, lat, lon)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []