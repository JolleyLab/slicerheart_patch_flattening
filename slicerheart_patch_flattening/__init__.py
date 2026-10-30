"""slicerheart_patch_flattening: 3D patch -> 2D template flattening for surgical planning.

Default method: ARAP (see pipeline.flatten_recommended and README "Recommendation").
"""
from .mesh import TriMesh
from .algorithms import flatten, METHODS
from .pipeline import flatten_recommended, DEFAULT_METHOD
__version__ = "0.2.0"
__all__ = ["TriMesh", "flatten", "METHODS", "flatten_recommended", "DEFAULT_METHOD"]
