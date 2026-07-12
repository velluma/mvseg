"""MVSeg — 3D TEE mitral valve segmentation."""

__version__ = "0.1.0"

# Class label semantics (index == label value in the GT volumes).
CLASS_NAMES = (
    "background",
    "anterior_leaflet",
    "posterior_leaflet",
    "mitral_valve_annulus",
    "aortic_valve_annulus",
)
NUM_CLASSES = len(CLASS_NAMES)
# Foreground classes only (exclude background) — used for metric reporting.
FG_CLASS_NAMES = CLASS_NAMES[1:]
