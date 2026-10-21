"""Presentation only (spec 10): a Panel app over the run store. Views call library functions of the store, the lab and
the viz package and render what they return; there is no computation of its own."""

from .app import build_app

__all__ = ["build_app"]
