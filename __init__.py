"""
Engine package for Cricket AI Analyzer.

This package contains all core analytics modules:
- Pose detection
- Shot segmentation
- Speed estimation
- Coaching feedback
- Highlight generation
- CSV logging

The central orchestrator is `CricketEngine`
defined in engine.py
"""

from engine.engine import CricketEngine

__all__ = ["CricketEngine"]
