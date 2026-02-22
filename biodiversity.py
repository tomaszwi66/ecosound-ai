"""
EcoSound AI - Ecosystem biodiversity assessment.
Interpretation of acoustic indices.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class EcosystemReport:
    """Full report on the acoustic state of an ecosystem."""
    aci: float = 0.0
    adi: float = 0.0
    bio: float = 0.0
    ndsi: float = 0.0
    spectral_entropy: float = 0.0
    temporal_entropy: float = 0.0
    num_sound_events: int = 0

    biodiversity_score: float = 0.0
    naturalness_score: float = 0.0
    acoustic_richness: float = 0.0
    overall_health: float = 0.0

    ecosystem_type: str = ""
    health_category: str = ""
    dominant_source: str = ""
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)

    band_analysis: dict = field(default_factory=dict)
    sound_events: list = field(default_factory=list)
    bird_species: list = field(default_factory=list)


class BiodiversityAssessor:
    """Assess biodiversity from acoustic indices."""

    REFERENCE = {
        'aci': {'poor': 0.1, 'moderate': 0.3, 'good': 0.5, 'excellent': 0.7},
        'adi': {'poor': 0.5, 'moderate': 1.0, 'good': 1.5, 'excellent': 2.0},
        'bio': {'poor': 5,   'moderate': 20,  'good': 50,  'excellent': 100},
    }

    def _normalize_score(self, value, poor, excellent):
        """Map *value* onto a 0–100 scale defined by *poor* and *excellent*."""
        score = (value - poor) / (excellent - poor + 1e-10) * 100
        return float(np.clip(score, 0, 100))

    def assess(self, aci, adi, bio, ndsi, spectral_ent, temporal_ent,
               sound_events, band_analysis, bird_species=None):
        """Run a full ecosystem assessment and return an EcosystemReport."""

        report = EcosystemReport(
            aci=aci, adi=adi, bio=bio, ndsi=ndsi,
            spectral_entropy=spectral_ent,
            temporal_entropy=temporal_ent,
            num_sound_events=len(sound_events),
            sound_events=sound_events,
            band_analysis=band_analysis,
            bird_species=bird_species or [],
        )

        # --- Biodiversity score ---
        aci_score   = self._normalize_score(
            aci, self.REFERENCE['aci']['poor'], self.REFERENCE['aci']['excellent'])
        adi_score   = self._normalize_score(
            adi, self.REFERENCE['adi']['poor'], self.REFERENCE['adi']['excellent'])
        event_score = min(len(sound_events) * 5, 100)

        # Bonus for identified bird species
        bird_bonus = 0
        if bird_species:
            bird_bonus = min(len(bird_species) * 8, 30)

        report.biodiversity_score = round(
            0.30 * aci_score + 0.30 * adi_score
            + 0.25 * event_score + 0.15 * bird_bonus * (100 / 30), 1)

        if bird_species:
            report.biodiversity_score = min(
                report.biodiversity_score + bird_bonus, 100)

        # --- Naturalness score ---
        report.naturalness_score = round(
            self._normalize_score(ndsi, -1.0, 1.0), 1)

        # --- Acoustic richness ---
        se_score = min(spectral_ent / 10.0 * 100, 100)
        te_score = min(temporal_ent / 8.0  * 100, 100)
        report.acoustic_richness = round(0.5 * se_score + 0.5 * te_score, 1)

        # --- Overall ecosystem health ---
        report.overall_health = round(
            0.35 * report.biodiversity_score
            + 0.30 * report.naturalness_score
            + 0.20 * report.acoustic_richness
            + 0.15 * self._normalize_score(
                bio, self.REFERENCE['bio']['poor'],
                self.REFERENCE['bio']['excellent']), 1)

        # --- Classification ---
        report.health_category = self._classify_health(report.overall_health)
        report.ecosystem_type  = self._guess_ecosystem(ndsi, band_analysis)
        report.dominant_source = self._dominant_source(ndsi)

        self._generate_warnings(report)
        self._generate_recommendations(report)

        return report

    def _classify_health(self, score):
        if score >= 80:
            return "🟢 EXCELLENT - Ecosystem in outstanding condition"
        elif score >= 60:
            return "🟡 GOOD - Healthy ecosystem with minor disturbances"
        elif score >= 40:
            return "🟠 MODERATE - Noticeable acoustic degradation"
        elif score >= 20:
            return "🔴 POOR - Significant ecosystem degradation"
        else:
            return "⚫ CRITICAL - Acoustically silent ecosystem"

    def _guess_ecosystem(self, ndsi, bands):
        if not bands:
            return "Unknown"

        high_bird = 0
        insect = 0
        for name, data in bands.items():
            p = data.get('proportion', 0)
            if '2-5 kHz' in name:
                high_bird = p
            elif '5-8 kHz' in name:
                insect = p

        if ndsi > 0.5 and high_bird > 20:
            return "🌲 Forest / Wilderness (bird-dominated)"
        elif ndsi > 0.3 and insect > 15:
            return "🌿 Meadow / Wetland (insects + birds)"
        elif ndsi > 0.0:
            return "🏞️ Semi-natural habitat"
        elif ndsi > -0.3:
            return "🏘️ Suburban / Urban park"
        else:
            return "🏙️ Urban / Industrial environment"

    def _dominant_source(self, ndsi):
        if ndsi > 0.5:
            return "🐦 Biophony (nature dominant)"
        elif ndsi > 0.0:
            return "🔀 Mixed (nature + anthropogenic)"
        elif ndsi > -0.3:
            return "🚗 Anthrophony (human activity dominant)"
        else:
            return "🏭 Strong anthrophony"

    def _generate_warnings(self, report):
        w = report.warnings
        if report.ndsi < -0.3:
            w.append("⚠️  Very high anthropogenic noise level!")
        if report.ndsi < 0:
            w.append("⚠️  Anthrophony exceeds biophony")
        if report.biodiversity_score < 30:
            w.append("⚠️  Low acoustic biodiversity")
        if report.num_sound_events < 5:
            w.append("⚠️  Few sound events detected - biological silence?")

    def _generate_recommendations(self, report):
        r = report.recommendations
        if report.ndsi < 0:
            r.append("🔇 Reduce anthropogenic noise sources")
            r.append("🌳 Create buffer zones (tree belts)")
        if report.biodiversity_score < 50:
            r.append("🐦 Install nest boxes for birds")
            r.append("🌻 Plant wildflower meadows for insects")
        if report.overall_health < 40:
            r.append("📊 Establish long-term acoustic monitoring")
        if report.overall_health >= 60:
            r.append("✅ Continue current conservation efforts")
            r.append("📅 Schedule periodic monitoring (monthly)")