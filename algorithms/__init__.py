# algorithms/__init__.py
"""Palette transfer algorithms for image color manipulation."""

from .kmeans_palette import KMeansReducedPalette, UniqueKMeansReducedPalette
from .reinhard_transfer import ReinhardColorTransfer
from .targeted_transfer import TargetedReinhardTransfer
from .skintone_transfer import SkinToneTransfer
from .entire_palette import EntirePalette

__all__ = [
    'KMeansReducedPalette',
    'UniqueKMeansReducedPalette',
    'ReinhardColorTransfer',
    'TargetedReinhardTransfer',
    'SkinToneTransfer',
    'EntirePalette',
]
