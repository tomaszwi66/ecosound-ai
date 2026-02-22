#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║                  🌍 EcoSound AI v1.0                       ║
║         Ecosystem Health Analysis through Sound            ║
╚══════════════════════════════════════════════════════════════╝

Usage:
  python main.py --demo                  Synthetic environment demo
  python main.py --file forest.wav       Analyse a file
  python main.py --file forest.wav --birds  + bird species ID
  python main.py --compare               Compare files in samples/
  python main.py --record                Record from microphone
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, resample

from audio_analyzer import AcousticIndices
from biodiversity import BiodiversityAssessor
from visualizer import create_full_dashboard


# ─── Global sample rate constant ─────────────────────────────────
TARGET_SR = 22050


def _resample_if_needed(audio, sr):
    """Return (audio, TARGET_SR). No-op if sample rate already matches."""
    if sr == TARGET_SR:
        return audio, TARGET_SR
    audio = resample(audio, int(len(audio) * TARGET_SR / sr))
    return audio, TARGET_SR


def _load_wav(filepath):
    """Load a WAV file and return mono audio resampled to TARGET_SR."""
    import soundfile as sf
    audio, sr = sf.read(str(filepath))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return _resample_if_needed(audio, sr)


# ─── Synthetic soundscape generator ──────────────────────────────

class EnvironmentSimulator:
    """Generates synthetic soundscapes for testing and demo purposes."""

    def __init__(self, sr=TARGET_SR, duration=30.0):
        self.sr = sr
        self.duration = duration
        self.n_samples = int(sr * duration)
        self.t = np.linspace(0, duration, self.n_samples)

    def _bird_song(self, t_start, duration=0.3,
                   base_freq=3000, complexity=3):
        """Synthesise a single bird vocalisation."""
        audio = np.zeros(self.n_samples)
        start = int(t_start * self.sr)
        length = int(duration * self.sr)
        end = min(start + length, self.n_samples)
        if start >= self.n_samples:
            return audio

        t_local = np.linspace(0, duration, end - start)
        for h in range(1, complexity + 1):
            freq = base_freq * h * 0.7
            fm = freq + 500 * np.sin(2 * np.pi * 15 * t_local) * h
            phase = 2 * np.pi * np.cumsum(fm) / self.sr
            envelope = np.sin(np.pi * t_local / duration) ** 2
            audio[start:end] += envelope * np.sin(phase) * (0.5 / h)
        return audio

    def _insect_chorus(self, freq=6000, intensity=0.1):
        """Synthesise a pulsing insect chorus around a centre frequency."""
        noise = np.random.randn(self.n_samples)
        nyq = self.sr / 2
        low = max((freq - 500) / nyq, 0.01)
        high = min((freq + 500) / nyq, 0.99)
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, noise)
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 8 * self.t)
        return filtered * pulse * intensity

    def _wind(self, intensity=0.05):
        """Synthesise low-frequency wind noise."""
        noise = np.random.randn(self.n_samples)
        b, a = butter(3, 200 / (self.sr / 2), btype='low')
        return filtfilt(b, a, noise) * intensity

    def _traffic(self, intensity=0.3):
        """Synthesise road traffic with random vehicle pass-bys."""
        noise = np.random.randn(self.n_samples)
        nyq = self.sr / 2
        b, a = butter(3, [100 / nyq, 1500 / nyq], btype='band')
        filtered = filtfilt(b, a, noise)
        for _ in range(np.random.randint(3, 8)):
            t_car = np.random.uniform(0, self.duration)
            car_env = np.exp(-0.5 * ((self.t - t_car) / 1.5) ** 2)
            filtered += car_env * np.random.randn(self.n_samples) * 0.1
        return filtered * intensity

    def _water_stream(self, intensity=0.08):
        """Synthesise broadband water stream noise."""
        noise = np.random.randn(self.n_samples)
        nyq = self.sr / 2
        b, a = butter(2, [300 / nyq, 3000 / nyq], btype='band')
        return filtfilt(b, a, noise) * intensity

    def pristine_forest(self):
        """🌲 Pristine forest soundscape."""
        audio = np.zeros(self.n_samples)
        species = [(2500, 3), (3500, 4), (4200, 2), (5000, 3),
                   (3000, 5), (6000, 2), (2800, 4), (4500, 3)]
        for freq, comp in species:
            for _ in range(np.random.randint(8, 20)):
                t_start = np.random.uniform(0, self.duration - 1)
                dur = np.random.uniform(0.1, 0.5)
                vol = np.random.uniform(0.05, 0.2)
                audio += self._bird_song(t_start, dur, freq, comp) * vol
        audio += self._insect_chorus(5500, 0.06)
        audio += self._insect_chorus(7000, 0.04)
        audio += self._wind(0.02)
        audio += self._water_stream(0.03)
        audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.8
        return audio, "🌲 Pristine Forest"

    def suburban_park(self):
        """🏞️ Suburban park soundscape."""
        audio = np.zeros(self.n_samples)
        for freq in [3000, 4000, 3500]:
            for _ in range(np.random.randint(3, 8)):
                t_start = np.random.uniform(0, self.duration - 1)
                audio += self._bird_song(t_start, 0.2, freq, 2) * 0.1
        audio += self._traffic(0.08)
        audio += self._wind(0.03)
        audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.7
        return audio, "🏞️ Suburban Park"

    def urban_center(self):
        """🏙️ Urban centre soundscape."""
        audio = np.zeros(self.n_samples)
        audio += self._traffic(0.4)
        for _ in range(np.random.randint(0, 3)):
            t_start = np.random.uniform(0, self.duration - 1)
            audio += self._bird_song(t_start, 0.15, 3000, 1) * 0.03
        audio += np.random.randn(self.n_samples) * 0.05
        audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.7
        return audio, "🏙️ Urban Centre"

    def meadow_wetland(self):
        """🌿 Meadow / wetland soundscape."""
        audio = np.zeros(self.n_samples)
        for freq in [1200, 1500, 1800]:
            for _ in range(np.random.randint(5, 15)):
                t_start = np.random.uniform(0, self.duration - 0.5)
                audio += self._bird_song(t_start, 0.1, freq, 1) * 0.12
        for freq in [2500, 3200]:
            for _ in range(np.random.randint(5, 12)):
                t_start = np.random.uniform(0, self.duration - 1)
                audio += self._bird_song(t_start, 0.3, freq, 3) * 0.08
        audio += self._insect_chorus(5000, 0.1)
        audio += self._insect_chorus(6500, 0.08)
        audio += self._insect_chorus(8000, 0.05)
        audio += self._water_stream(0.05)
        audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.8
        return audio, "🌿 Meadow / Wetland"


# ─── Analysis ─────────────────────────────────────────────────────

def analyze_audio(audio, sr, label="Audio", show_dashboard=True,
                  save_path=None, use_birdnet=False,
                  lat=52.0, lon=21.0):
    """Run a full EcoSound AI analysis on an audio array."""

    print(f"\n{'═' * 60}")
    print(f"  🎵 Analysing: {label}")
    print(f"  ⏱  Duration: {len(audio)/sr:.1f}s | SR: {sr} Hz")
    print(f"{'═' * 60}")

    analyzer = AcousticIndices(sample_rate=sr)
    assessor = BiodiversityAssessor()

    print("  📐 Computing acoustic indices...")
    aci      = analyzer.acoustic_complexity_index(audio)
    adi      = analyzer.acoustic_diversity_index(audio)
    bio      = analyzer.bioacoustic_index(audio)
    ndsi_val = analyzer.ndsi(audio)
    se       = analyzer.spectral_entropy(audio)
    te       = analyzer.temporal_entropy(audio)
    events   = analyzer.detect_sound_events(audio)
    bands    = analyzer.frequency_band_analysis(audio)

    print(f"    ACI  = {aci:.4f}")
    print(f"    ADI  = {adi:.4f}")
    print(f"    BIO  = {bio:.2f}")
    print(f"    NDSI = {ndsi_val:+.4f}")
    print(f"    H(s) = {se:.4f}")
    print(f"    H(t) = {te:.4f}")
    print(f"    Sound events: {len(events)}")

    # Optional bird species identification
    bird_species = []
    if use_birdnet:
        from bird_identifier import identify_birds
        bird_species = identify_birds(audio, sr, lat=lat, lon=lon)

    print("\n  🧠 Assessing biodiversity...")
    report = assessor.assess(
        aci, adi, bio, ndsi_val, se, te, events, bands, bird_species
    )

    print(f"\n  {'─' * 50}")
    print(f"  📊 RESULTS:")
    print(f"    Biodiversity:      {report.biodiversity_score:.1f}/100")
    print(f"    Naturalness:       {report.naturalness_score:.1f}/100")
    print(f"    Acoustic richness: {report.acoustic_richness:.1f}/100")
    print(f"    ECOSYSTEM HEALTH:  {report.overall_health:.1f}/100")
    print(f"  {'─' * 50}")
    print(f"  {report.health_category}")
    print(f"  {report.ecosystem_type}")
    print(f"  {report.dominant_source}")

    if report.bird_species:
        print(f"\n  🐦 Identified species: {len(report.bird_species)}")
        for bird in report.bird_species[:5]:
            name = bird.get('common_name', '?')
            conf = bird.get('confidence', 0)
            print(f"    • {name} ({conf:.0%})")

    if report.warnings:
        print(f"\n  ⚠️  WARNINGS:")
        for w in report.warnings:
            print(f"    {w}")

    if report.recommendations:
        print(f"\n  💡 RECOMMENDATIONS:")
        for r in report.recommendations:
            print(f"    {r}")

    if show_dashboard:
        print("\n  📊 Rendering dashboard...")
        create_full_dashboard(audio, sr, report, save_path)

    return report


def compare_locations():
    """Compare all .wav files found in the samples/ directory."""

    samples_dir = Path("samples")
    if not samples_dir.exists():
        samples_dir.mkdir()
        print("  📁 Created 'samples/' directory")
        print("  Place .wav files there and run again")
        return

    wav_files = sorted(samples_dir.glob("*.wav"))
    if not wav_files:
        print("  📁 'samples/' directory is empty")
        print("  Drop some .wav recordings there and try again")
        return

    print(f"\n  🔍 Found {len(wav_files)} file(s):")
    for f in wav_files:
        print(f"    📄 {f.name}")

    # Load each file once and keep audio in memory
    loaded = {}
    for filepath in wav_files:
        try:
            audio, sr = _load_wav(filepath)
            loaded[filepath.name] = (audio, sr)
        except Exception as e:
            print(f"  ❌ Failed to load {filepath.name}: {e}")

    results = {}
    for name, (audio, sr) in loaded.items():
        try:
            report = analyze_audio(
                audio, sr, name, show_dashboard=False
            )
            results[name] = report
        except Exception as e:
            print(f"  ❌ Analysis failed for {name}: {e}")

    if len(results) < 2:
        return

    # Comparison table
    print(f"\n\n{'═' * 72}")
    print(f"  📊 LOCATION COMPARISON")
    print(f"{'═' * 72}")
    print(
        f"  {'File':<25} {'ACI':>6} {'ADI':>6} "
        f"{'NDSI':>7} {'BIO':>5} {'Health':>8}"
    )
    print(f"  {'─'*25} {'─'*6} {'─'*6} {'─'*7} {'─'*5} {'─'*8}")

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].overall_health,
        reverse=True
    )

    for name, r in sorted_results:
        print(
            f"  {name:<25} {r.aci:>6.3f} {r.adi:>6.3f} "
            f"{r.ndsi:>+7.3f} {r.bio:>5.1f} {r.overall_health:>7.1f}"
        )

    # Ranking
    print(f"\n  🏆 RANKING:")
    medals = ['🥇', '🥈', '🥉']
    for i, (name, r) in enumerate(sorted_results):
        medal = medals[i] if i < 3 else f'  {i+1}.'
        print(f"  {medal} {name}")
        print(f"      {r.health_category}")

    # Score spread
    best  = sorted_results[0][1].overall_health
    worst = sorted_results[-1][1].overall_health
    diff  = best - worst
    print(f"\n  📈 Score spread: {diff:.1f} pts", end=' ')
    if diff > 30:
        print("→ Significant")
    elif diff > 15:
        print("→ Moderate")
    else:
        print("→ Minor")

    # Dashboard for best and worst location (reuse already-loaded audio)
    print(f"\n  📊 Generating dashboards for best and worst location...")
    for name, r in [sorted_results[0], sorted_results[-1]]:
        audio, sr = loaded[name]
        save_name = Path(name).stem + '_report.png'
        create_full_dashboard(audio, sr, r, save_name)


# ─── Entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='🌍 EcoSound AI - Ecosystem health analysis through sound',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                      Run synthetic demo
  python main.py --demo --env forest         Demo: forest only
  python main.py --file recording.wav        Analyse a file
  python main.py --file recording.wav --birds  + bird species ID
  python main.py --compare                   Compare files in samples/
  python main.py --record                    Record from microphone
  python main.py --record --duration 60      Record for 60 seconds
        """
    )

    parser.add_argument('--file', '-f', type=str,
                        help='Path to a .wav file')
    parser.add_argument('--demo', '-d', action='store_true',
                        help='Run synthetic demo')
    parser.add_argument('--env', '-e',
                        choices=['forest', 'park', 'city', 'meadow', 'all'],
                        default='all',
                        help='Demo environment')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare .wav files in samples/')
    parser.add_argument('--record', '-r', action='store_true',
                        help='Record from microphone')
    parser.add_argument('--birds', '-b', action='store_true',
                        help='Enable bird ID (requires BirdNET)')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save dashboard to PNG')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Recording / demo duration in seconds')
    parser.add_argument('--lat', type=float, default=52.0,
                        help='Latitude for BirdNET localisation')
    parser.add_argument('--lon', type=float, default=21.0,
                        help='Longitude for BirdNET localisation')

    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           🌍  E c o S o u n d   A I   v 1.0            ║
    ║        Ecosystem Health Analysis through Sound          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # ─── File analysis ───────────────────────────
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"  ❌ File not found: {filepath}")
            sys.exit(1)

        audio, sr = _load_wav(filepath)
        save_path = (str(filepath.stem) + '_report.png'
                     if args.save else None)

        analyze_audio(
            audio, sr, filepath.name,
            save_path=save_path,
            use_birdnet=args.birds,
            lat=args.lat, lon=args.lon
        )

    # ─── Microphone recording ────────────────────
    elif args.record:
        from recorder import record, check_microphone
        if not check_microphone():
            sys.exit(1)

        os.makedirs("samples", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"samples/ecosound_{timestamp}.wav"

        audio, sr = record(args.duration, TARGET_SR, filename)
        save_path = f"samples/report_{timestamp}.png" if args.save else None

        analyze_audio(
            audio, sr, f"Recording {timestamp}",
            save_path=save_path,
            use_birdnet=args.birds,
            lat=args.lat, lon=args.lon
        )

    # ─── Location comparison ─────────────────────
    elif args.compare:
        compare_locations()

    # ─── Synthetic demo ──────────────────────────
    elif args.demo:
        sim = EnvironmentSimulator(sr=TARGET_SR, duration=args.duration)

        environments = {
            'forest': sim.pristine_forest,
            'park':   sim.suburban_park,
            'city':   sim.urban_center,
            'meadow': sim.meadow_wetland,
        }

        envs = environments if args.env == 'all' \
               else {args.env: environments[args.env]}

        results = {}
        for name, gen_func in envs.items():
            audio, label = gen_func()
            save_path = f'ecosound_{name}.png' if args.save else None
            report = analyze_audio(audio, TARGET_SR, label,
                                   save_path=save_path)
            results[label] = report

        if len(results) > 1:
            print(f"\n\n{'═' * 60}")
            print(f"  📊 ENVIRONMENT COMPARISON")
            print(f"{'═' * 60}")
            sorted_r = sorted(
                results.items(),
                key=lambda x: x[1].overall_health,
                reverse=True
            )
            for label, r in sorted_r:
                print(
                    f"  {label:<30} "
                    f"Health: {r.overall_health:.0f}/100  "
                    f"NDSI: {r.ndsi:+.3f}"
                )

    # ─── No arguments: quick demo ────────────────
    else:
        print("  Usage:")
        print("    python main.py --demo             Quick demo")
        print("    python main.py --file rec.wav     Analyse a file")
        print("    python main.py --compare          Compare samples/")
        print("    python main.py --record           Record from mic")
        print("    python main.py --help             Full help")
        print()
        print("  ▶  Running quick demo...\n")

        sim = EnvironmentSimulator(sr=TARGET_SR, duration=15.0)
        for gen_func in [sim.pristine_forest, sim.urban_center]:
            audio, label = gen_func()
            analyze_audio(audio, TARGET_SR, label)


if __name__ == '__main__':
    main()