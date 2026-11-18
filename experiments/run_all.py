"""Run every experiment in order (about 3-4 minutes on one CPU core)."""
import pathlib
import runpy
import time

HERE = pathlib.Path(__file__).parent
for script in sorted(HERE.glob("e0*.py")):
    t = time.time()
    print(f"== {script.name}")
    runpy.run_path(str(script), run_name="__main__")
    print(f"   ({time.time() - t:.1f}s)")
