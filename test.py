import sys
import os
import re
import inspect
import traceback
from pathlib import Path

print("=" * 72)
print("PEARL / OPEN-PEARL-MINER FULL DIAGNOSTIC")
print("=" * 72)

# ------------------------------------------------------------------
# 1. Environment
# ------------------------------------------------------------------

print("\n[1] ENVIRONMENT")
print("Python:", sys.version)
print("Executable:", sys.executable)
print("PWD:", os.getcwd())

try:
    import pearl_mining as pm
    print("pearl_mining:", pm.__file__)
except Exception:
    print("FAILED to import pearl_mining")
    traceback.print_exc()
    sys.exit(1)

try:
    import numpy
    print("numpy:", numpy.__version__, numpy.__file__)
except Exception as e:
    print("numpy import failed:", e)

try:
    import blake3
    print("blake3:", blake3.__file__)
except Exception as e:
    print("blake3 import failed:", e)


# ------------------------------------------------------------------
# 2. Consensus API
# ------------------------------------------------------------------

print("\n[2] PEARL CONSENSUS API")

print("PENALTY_BASE_RANK:",
      getattr(pm, "PENALTY_BASE_RANK", "<missing>"))

for name in [
    "MiningConfiguration",
    "MoEConfig",
    "penalized_target_bound",
]:
    obj = getattr(pm, name, None)

    print(f"\n{name}: {obj}")

    if obj is not None:
        try:
            print(" signature:", inspect.signature(obj))
        except Exception as e:
            print(" signature unavailable:", e)

        try:
            attrs = [
                x for x in dir(obj)
                if not x.startswith("_")
            ]
            print(" attrs:", attrs)
        except Exception:
            pass


# ------------------------------------------------------------------
# 3. Dump potentially relevant API symbols
# ------------------------------------------------------------------

print("\n[3] RELEVANT pearl_mining SYMBOLS")

keywords = [
    "mma",
    "pattern",
    "rank",
    "target",
    "penal",
    "config",
    "tile",
]

for name in sorted(dir(pm)):
    if any(k in name.lower() for k in keywords):
        obj = getattr(pm, name)
        print(f"{name}: {obj}")

        try:
            print("   signature:", inspect.signature(obj))
        except Exception:
            pass


# ------------------------------------------------------------------
# 4. Read Open-Pearl-Miner Python source
# ------------------------------------------------------------------

print("\n[4] OPEN-PEARL-MINER SOURCE INSPECTION")

root = Path.cwd()
python_dir = root / "python"

patterns = [
    r"MiningConfiguration",
    r"mma_type",
    r"rows_pattern",
    r"cols_pattern",
    r"noise_rank",
    r"rounded_common_dim",
    r"hash_tile",
    r"factor\s*=",
    r"bound\s*=",
    r"penalized_target",
    r"PlainProof",
]

for file in sorted(python_dir.glob("*.py")):
    try:
        lines = file.read_text(errors="replace").splitlines()
    except Exception:
        continue

    matches = []

    for i, line in enumerate(lines, 1):
        if any(re.search(p, line, re.I) for p in patterns):
            matches.append((i, line))

    if matches:
        print(f"\n--- {file} ---")
        for i, line in matches:
            print(f"{i:5d}: {line}")


# ------------------------------------------------------------------
# 5. Import miner modules and inspect configuration objects
# ------------------------------------------------------------------

print("\n[5] MINER MODULE INSPECTION")

for module_name in ["miner_capi", "pearl_host"]:
    try:
        mod = __import__(module_name)
        print(f"\nImported {module_name}: {mod.__file__}")

        for name in dir(mod):
            if any(k in name.lower() for k in [
                "config", "mma", "pattern", "rank", "tile"
            ]):
                try:
                    value = getattr(mod, name)
                    if not inspect.ismodule(value):
                        print(f"  {name} = {value}")
                except Exception:
                    pass

    except Exception as e:
        print(f"\nCould not import {module_name}: {e}")


# ------------------------------------------------------------------
# 6. Known runtime values from actual RTX 4090 run
# ------------------------------------------------------------------

print("\n[6] VALUES OBSERVED FROM ACTUAL MINER RUN")

target = 1 << 205
factor = 1 << 20
old_bound = min(target * factor, (1 << 256) - 1)

print("target:")
print("  value       =", target)
print("  log2        = 205")

print("factor:")
print("  value       =", factor)
print("  log2        = 20")

print("old miner bound:")
print("  value       =", old_bound)
print("  approx log2 = 225")
print("  bit_length  =", old_bound.bit_length())

print("\nObserved GPU:")
print("  RTX 4090 / SM89")
print("  ~172-174 TH/s")
print("  repeated candidate hits")
print("  pool rejects with code 23")
print("  'Jackpot condition not satisfied:")
print("   hash does not meet the rank-penalized difficulty target'")


# ------------------------------------------------------------------
# 7. Try common MiningConfiguration constructions automatically
# ------------------------------------------------------------------

print("\n[7] AUTOMATIC MiningConfiguration TESTS")

MC = getattr(pm, "MiningConfiguration", None)
penalty_fn = getattr(pm, "penalized_target_bound", None)

if MC is None or penalty_fn is None:
    print("Cannot test: API missing")
else:
    # Look for enum-like objects exported by pearl_mining
    candidates = {}

    for name in dir(pm):
        obj = getattr(pm, name)

        if any(k in name.lower() for k in [
            "mma", "pattern"
        ]):
            candidates[name] = obj

    print("Potential enum/type candidates:")

    for name, obj in candidates.items():
        print(f"\n{name}: {obj}")

        try:
            for attr in dir(obj):
                if not attr.startswith("_"):
                    try:
                        val = getattr(obj, attr)
                        if not callable(val):
                            print(f"  {attr} = {val}")
                    except Exception:
                        pass
        except Exception:
            pass


# ------------------------------------------------------------------
# 8. Inspect MiningConfiguration annotations/repr/help
# ------------------------------------------------------------------

print("\n[8] MiningConfiguration DETAILS")

if MC is not None:
    try:
        print("repr:", MC)
        print("signature:", inspect.signature(MC))
    except Exception:
        pass

    try:
        print("doc:", MC.__doc__)
    except Exception:
        pass

    try:
        print("text signature:", MC.__text_signature__)
    except Exception:
        pass


# ------------------------------------------------------------------
# 9. Search installed pearl_mining package for useful strings
# ------------------------------------------------------------------

print("\n[9] INSTALLED PACKAGE LOCATION")

pkg = Path(pm.__file__).resolve().parent
print("package directory:", pkg)

for f in pkg.rglob("*"):
    if f.is_file() and f.suffix in [".py", ".pyi"]:
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue

        if (
            "MiningConfiguration" in text
            or "penalized_target_bound" in text
        ):
            print(f"\n--- {f} ---")

            for i, line in enumerate(text.splitlines(), 1):
                if (
                    "MiningConfiguration" in line
                    or "penalized_target_bound" in line
                    or "Mma" in line
                    or "Pattern" in line
                ):
                    print(f"{i:5d}: {line}")


print("\n" + "=" * 72)
print("DIAGNOSTIC COMPLETE")
print("=" * 72)
