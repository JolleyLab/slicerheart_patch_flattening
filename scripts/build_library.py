"""Build and write the synthetic patch library.  python scripts/build_library.py [out_dir]"""
import sys
from slicerheart_patch_flattening.library import build_library, write_library

out = sys.argv[1] if len(sys.argv) > 1 else "output/synthetic_library"
man = write_library(out, build_library())
print(f"wrote {len(man)} shapes to {out}")
