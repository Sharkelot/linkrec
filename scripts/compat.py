"""Compatibility shims for RecBole 1.2.0 / PyG 2.8 on this host's modern stack
(Python 3.14, numpy 2.x, scipy 1.18, torch 2.14).

Import once at the top of any entry point, before importing recbole:

    from compat import install_shims
    install_shims()

See skill `recbole-pyg-modern-stack` for the full rationale.
"""
import functools
import warnings


def install_shims() -> None:
    import numpy as np

    # RecBole 1.2.0 compatibility_settings() touches aliases removed in NumPy 2.0.
    # np.object_/np.unicode_ must be CLASSES (pandas does issubclass checks).
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
        np.float = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128  # type: ignore[attr-defined]
        np.complex = np.complex128  # type: ignore[attr-defined]
    if not hasattr(np, "object_"):
        np.object_ = object  # type: ignore[attr-defined]
        np.object = object  # type: ignore[attr-defined]
    if not hasattr(np, "unicode_"):
        np.unicode_ = str  # type: ignore[attr-defined]
        np.unicode = str  # type: ignore[attr-defined]

    # scipy >= 1.14 removed the private dok_matrix._update used by
    # LightGCN.get_norm_adj_mat.
    import scipy.sparse as sp
    if not hasattr(sp.dok_matrix, "_update"):
        sp.dok_matrix._update = lambda self, other: self.update(other)  # type: ignore[attr-defined]

    # torch 2.6+ defaults torch.load to weights_only=True; RecBole's trainer
    # internally reloads its own (trusted, just-saved) checkpoints.
    import torch
    if getattr(torch.load, "_linkrec_patched", False):
        return
    orig_load = torch.load

    @functools.wraps(orig_load)
    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    patched_load._linkrec_patched = True  # type: ignore[attr-defined]
    torch.load = patched_load  # type: ignore[assignment]


def quiet_recbole_noise() -> None:
    """Silence RecBole/pandas deprecation noise that is harmless here."""
    warnings.filterwarnings("ignore")
