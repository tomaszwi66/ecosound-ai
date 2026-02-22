"""
EcoSound AI - Microphone recording.
"""

import numpy as np
import sys
import os
import time


def check_microphone():
    """Check whether a microphone input device is available."""
    try:
        import sounddevice as sd
        default = sd.query_devices(kind='input')
        print(f"  ✅ Microphone: {default['name']}")
        print(f"     Sample rate: {int(default['default_samplerate'])} Hz")
        return True
    except ImportError:
        print("  ❌ sounddevice not found: pip install sounddevice")
        return False
    except Exception as e:
        print(f"  ❌ Microphone error: {e}")
        return False


def record(duration=30.0, sr=22050, filename=None):
    """Record audio from the microphone and optionally save to a WAV file."""
    import sounddevice as sd

    print(f"\n  🎙️  RECORDING ({duration}s)")

    print("  Starting in: ", end='')
    for i in range(3, 0, -1):
        print(f"{i}... ", end='', flush=True)
        time.sleep(1)

    print("\n  🔴 RECORDING - keep still!")

    audio = sd.rec(int(duration * sr), samplerate=sr,
                   channels=1, dtype='float32')

    total = int(duration)
    for step in range(total):
        time.sleep(1)
        progress = (step + 1) / total
        filled = int(30 * progress)
        bar = '█' * filled + '░' * (30 - filled)
        remaining = total - step - 1
        sys.stdout.write(
            f'\r  [{bar}] {progress*100:.0f}% ({remaining}s remaining)'
        )
        sys.stdout.flush()

    sd.wait()
    print(f"\n  ⬜ DONE!")

    audio = audio.flatten()
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    if filename:
        import soundfile as sf
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        sf.write(filename, audio, sr)
        size_mb = len(audio) * 4 / 1024 / 1024
        print(f"  💾 Saved: {filename} ({size_mb:.1f} MB)")

    return audio, sr