from pathlib import Path

dir = Path(__file__).parent / 'assets' / 'meshes' / 'phantom3'

for hull in dir.iterdir():
    stem = hull.stem
    if "hull" in stem:
        new_name = stem.replace("phantom3_", "")
    else:
        new_name = stem.replace("phantom3", "visual")
    hull.rename(hull.parent / (new_name + hull.suffix))
