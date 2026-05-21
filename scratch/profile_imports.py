import sys
import time
import glob
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "monte_carlo"))
sys.path.insert(0, str(REPO_ROOT / "dynamics"))
sys.path.insert(0, str(REPO_ROOT / "control_layer"))

test_files = glob.glob(str(REPO_ROOT / "tests" / "test_*.py"))
test_files.sort()

print(f"Profiling {len(test_files)} test files import times...")

import_times = []
for f in test_files:
    filename = Path(f).name
    # Measure time
    start = time.perf_counter()
    try:
        # We can dynamically import the module
        module_name = filename[:-3]  # remove .py
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # We need to add tests directory to sys.path to import
        if str(REPO_ROOT / "tests") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "tests"))
            
        __import__(module_name)
        elapsed = time.perf_counter() - start
        import_times.append((filename, elapsed))
    except BaseException as e:
        elapsed = time.perf_counter() - start
        import_times.append((filename, elapsed, f"Skipped or error: {type(e).__name__}"))

# Sort by time descending
import_times.sort(key=lambda x: x[1], reverse=True)

print("\nTop 15 slowest test files to import:")
for item in import_times[:15]:
    if len(item) == 2:
        print(f"  {item[0]}: {item[1]:.4f}s")
    else:
        print(f"  {item[0]}: {item[1]:.4f}s ({item[2]})")
