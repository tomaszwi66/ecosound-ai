"""
EcoSound AI - Visualisations.
Single window, non-blocking, fast.
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal


def _set_backend():
    """
    Auto-select a matplotlib backend:
    - If a GUI backend is already active, leave it untouched.
    - Otherwise try TkAgg, Qt5Agg, QtAgg, WXAgg in order.
    - Fall back to Agg (file-only, no GUI window) if nothing works.
    """
    if matplotlib.get_backend().lower() not in ('agg', ''):
        return

    for backend in ['TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg']:
        try:
            matplotlib.use(backend, force=False)
            import matplotlib.pyplot as _plt  # noqa: F401
            return
        except Exception:
            continue

    matplotlib.use('Agg')


_set_backend()

COLORS = {
    'bg':     '#0a0a12',
    'panel':  '#12122a',
    'panel2': '#16162e',
    'text':   '#d0d0e8',
    'dim':    '#6060a0',
    'accent': '#00d4aa',
    'blue':   '#4ecdc4',
    'purple': '#a855f7',
    'orange': '#f59e0b',
    'green':  '#22c55e',
    'red':    '#ef4444',
}

BAR_COLORS = ['#ef4444', '#f59e0b', '#22c55e',
               '#3b82f6', '#a855f7', '#ec4899']

# Display labels for frequency band keys returned by band_analysis
BAND_LABELS = {
    'geofonia (< 200 Hz)':              'Geo <200',
    'ssaki, antropofonia (200-1000 Hz)': 'Anthro',
    'żaby, duże ptaki (1-2 kHz)':       'Frogs',
    'ptaki śpiewające (2-5 kHz)':       'Birds',
    'owady, małe ptaki (5-8 kHz)':      'Insects',
    'nietoperze, owady (> 8 kHz)':      'Ultra',
}


def _style_ax(ax):
    """Apply shared dark-theme styling to a regular (non-polar) axis."""
    ax.set_facecolor(COLORS['panel'])
    ax.tick_params(colors=COLORS['text'], labelsize=6)
    ax.spines[:].set_edgecolor(COLORS['dim'])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def create_full_dashboard(audio, sr, report, output_path=None):
    """
    Render the full EcoSound AI dashboard in a single figure.

    If *output_path* is given, save to PNG and skip the GUI window.
    If *output_path* is None, open an interactive window (requires a
    GUI backend such as TkAgg or Qt5Agg; falls back to saving a PNG
    named 'ecosound_dashboard.png' when only Agg is available).
    """

    plt.rcParams['figure.dpi'] = 96   # standard laptop DPI
    fig = plt.figure(figsize=(11, 6.5), facecolor=COLORS['bg'])
    fig.suptitle(
        'EcoSound AI',
        fontsize=10, color=COLORS['accent'],
        fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.60, wspace=0.38,
        left=0.08, right=0.97,
        top=0.92, bottom=0.08
    )

    # Row 1: spectrogram (full width)
    ax_spec = fig.add_subplot(gs[0, :])
    _plot_spectrogram(ax_spec, audio, sr)

    # Row 2: waveform (full width)
    ax_wave = fig.add_subplot(gs[1, :])
    _plot_waveform(ax_wave, audio, sr, report)

    # Row 3: frequency bands | radar profile | scores
    ax_bands  = fig.add_subplot(gs[2, 0])
    ax_radar  = fig.add_subplot(gs[2, 1], polar=True)
    ax_scores = fig.add_subplot(gs[2, 2])

    _plot_bands(ax_bands, report)
    _plot_radar(ax_radar, report)
    _plot_scores(ax_scores, report)

    # Summary text strip below the main title
    _draw_summary_text(fig, report)

    # Save or display
    if output_path:
        fig.savefig(output_path, dpi=100,
                    facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"  📊 Saved: {output_path}")
        plt.close(fig)
    else:
        backend = matplotlib.get_backend().lower()
        if backend == 'agg':
            fallback = 'ecosound_dashboard.png'
            fig.savefig(fallback, dpi=100,
                        facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
            print(
                f"  ⚠️  No GUI environment available (backend=Agg).\n"
                f"  📊 Dashboard saved to: {fallback}\n"
                f"  💡 To open a window: install tkinter "
                f"(sudo apt install python3-tk) and run again."
            )
        else:
            plt.show()


# ─── Panel renderers ──────────────────────────────────────────────────────────

def _plot_spectrogram(ax, audio, sr):
    """Render a fast power spectrogram, downsampled if necessary."""
    _style_ax(ax)

    # Downsample to at most 60 000 samples for speed
    target = 60_000
    if len(audio) > target:
        step = len(audio) // target
        audio_down = audio[::step]
        sr_down = sr // step
    else:
        audio_down, sr_down = audio, sr

    freqs, times, Sxx = signal.spectrogram(
        audio_down, fs=sr_down,
        nperseg=512, noverlap=384   # smaller window = faster
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    ax.pcolormesh(times, freqs, Sxx_db,
                  shading='nearest',   # faster than 'gouraud'
                  cmap='magma', rasterized=True)

    for freq, color, label in [
        (200,  'red',    '200 Hz'),
        (2000, 'yellow', '2 kHz'),
        (8000, 'lime',   '8 kHz'),
    ]:
        ax.axhline(y=freq, color=color, alpha=0.45,
                   linewidth=0.8, linestyle='--')
        ax.text(times[-1] * 0.99, freq * 1.05, label,
                color=color, fontsize=5, alpha=0.7, ha='right')

    ax.set_ylabel('Hz', color=COLORS['text'], fontsize=5)
    ax.set_title('Spectrogram', color=COLORS['accent'], fontsize=7)
    ax.set_ylim(0, min(sr_down // 2, 11_000))


def _plot_waveform(ax, audio, sr, report):
    """Render the audio waveform with detected sound events highlighted."""
    _style_ax(ax)

    time_axis = np.arange(len(audio)) / sr
    step = max(1, len(audio) // 8_000)   # cap at 8 000 points

    ax.plot(time_axis[::step], audio[::step],
            color=COLORS['blue'], alpha=0.55, linewidth=0.35,
            rasterized=True)

    for ev in report.sound_events[:15]:
        ax.axvspan(ev['start_time'], ev['end_time'],
                   alpha=0.18, color=COLORS['green'],
                   linewidth=0)

    ax.set_xlabel('Time [s]', color=COLORS['text'], fontsize=5)
    ax.set_title(
        f'Waveform  |  Sound events: {len(report.sound_events)}',
        color=COLORS['accent'], fontsize=7
    )


def _plot_bands(ax, report):
    """Render a horizontal bar chart of energy per frequency band."""
    _style_ax(ax)
    ax.set_facecolor(COLORS['panel2'])

    if report.band_analysis:
        names = [BAND_LABELS.get(k, k[:10])
                 for k in report.band_analysis]
        props = [d['proportion']
                 for d in report.band_analysis.values()]

        bars = ax.barh(names, props,
                       color=BAR_COLORS[:len(names)],
                       alpha=0.85, height=0.45)

        for bar, val in zip(bars, props):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%',
                    ha='left', va='center',
                    color=COLORS['text'], fontsize=5)
        ax.invert_yaxis()

    ax.set_xlabel('%', color=COLORS['text'], fontsize=5)
    ax.set_title('Frequency Bands', color=COLORS['accent'], fontsize=7)


def _plot_radar(ax, report):
    """Render a radar / spider chart of the five acoustic indices."""
    ax.set_facecolor(COLORS['bg'])
    ax.tick_params(colors=COLORS['text'], labelsize=6)

    cats = ['Bio', 'Nat.', 'Rich.', 'ACI', 'ADI']
    vals = [
        report.biodiversity_score,
        report.naturalness_score,
        report.acoustic_richness,
        min(report.aci * 150, 100),
        min(report.adi * 45,  100),
    ]

    angles = np.linspace(0, 2 * np.pi, len(cats),
                         endpoint=False).tolist()
    vals_plot   = vals + vals[:1]
    angles_plot = angles + angles[:1]

    ax.plot(angles_plot, vals_plot,
            color=COLORS['accent'], linewidth=1.8)
    ax.fill(angles_plot, vals_plot,
            color=COLORS['accent'], alpha=0.22)

    ax.set_xticks(angles)
    ax.set_xticklabels(cats, fontsize=6, color=COLORS['text'])
    ax.set_ylim(0, 100)
    ax.set_yticks([50, 100])
    ax.set_yticklabels(['50', '100'], fontsize=4,
                       color=COLORS['dim'])
    ax.grid(color=COLORS['dim'], alpha=0.3)
    ax.set_title('Profile', color=COLORS['accent'],
                 fontsize=7, pad=8)
    ax.spines['polar'].set_edgecolor(COLORS['dim'])


def _plot_scores(ax, report):
    """Render a horizontal bar chart of the four main health scores."""
    _style_ax(ax)
    ax.set_facecolor(COLORS['panel2'])

    names  = ['HEALTH', 'Biodiv.', 'Natural.', 'Richness']
    values = [
        report.overall_health,
        report.biodiversity_score,
        report.naturalness_score,
        report.acoustic_richness,
    ]
    colors = [COLORS['accent'], COLORS['blue'],
              COLORS['green'],  COLORS['purple']]

    bars = ax.barh(names, values,
                   color=colors, alpha=0.85, height=0.4)

    for bar, val in zip(bars, values):
        label_color = (COLORS['green']  if val >= 60 else
                       COLORS['orange'] if val >= 40 else
                       COLORS['red'])
        ax.text(bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}/100',
                ha='left', va='center',
                color=label_color, fontsize=6, fontweight='bold')

    ax.set_xlim(0, 115)
    ax.set_title('Scores', color=COLORS['accent'], fontsize=7)
    ax.invert_yaxis()

    for x, color in [(40, COLORS['red']),
                     (60, COLORS['orange']),
                     (80, COLORS['green'])]:
        ax.axvline(x=x, color=color, alpha=0.25,
                   linewidth=0.6, linestyle=':')


def _draw_summary_text(fig, report):
    """Draw a compact one-line summary strip just below the main title."""
    parts = [
        report.health_category,
        report.ecosystem_type,
        f"ACI={report.aci:.3f}  ADI={report.adi:.3f}  "
        f"NDSI={report.ndsi:+.3f}",
        f"BIO={report.bio:.1f}  Events={report.num_sound_events}",
    ]
    if report.bird_species:
        bird_list = ', '.join(
            b.get('common_name', '?')
            for b in report.bird_species[:3]
        )
        parts.append(f"Birds: {bird_list}")
    if report.warnings:
        parts.extend(report.warnings[:2])

    text = '   |   '.join(filter(None, parts))

    fig.text(0.50, 0.955, text,
             fontsize=5.5, color=COLORS['dim'],
             ha='center', va='top',
             family='monospace',
             wrap=True)