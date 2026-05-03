import json
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

def verify_phase0():
    print("--- Phase 0 Verification ---")
    
    # 1. Check docs
    docs = ["docs/comparison_table.md", "docs/TECHNICAL_SPEC.md"]
    for d in docs:
        if Path(d).exists():
            print(f"[OK] {d} exists")
        else:
            print(f"[FAIL] {d} missing")

    # 2. Check profiles
    with open("anchor_profiles.json", "r") as f:
        profiles_data = json.load(f)
    profiles = profiles_data["profiles"] if isinstance(profiles_data, dict) else profiles_data
    profile_names = [p["name"] for p in profiles]
    if "paper-recommended" in profile_names:
        print("[OK] 'paper-recommended' profile found")
    else:
        print("[FAIL] 'paper-recommended' profile missing")

    # 3. Check claims
    with open("results/anchor_claims.json", "r") as f:
        claims_data = json.load(f)
    
    # Handle both list and dict-with-anchor_claims formats
    if isinstance(claims_data, dict) and "anchor_claims" in claims_data:
        claims = claims_data["anchor_claims"]
    else:
        claims = claims_data
        
    claim_profiles = [c["profile"] for c in claims]
    if "paper-recommended" in claim_profiles:
        print("[OK] 'paper-recommended' claim found")
    else:
        print("[FAIL] 'paper-recommended' claim missing")

    # 4. Check stress defaults
    from src.sgms_anchor_v1 import mission_level_metrics
    # Calling with no jacket_material should now default to CFRP and be feasible
    # Sobol-optimal GdBCO is not feasible with BFRP (SF=1.04 < 1.5)
    r = mission_level_metrics(
        u=588.8, mp=4.57, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.0004, k_fp=6000,
        magnet_material="GdBCO", spacing=0.48
    )
    # GdBCO @ 50k RPM stress is 765 MPa. 
    # BFRP limit is 800 MPa / 1.5 = 533 MPa -> margin = 0.70 < 1.0 (fail)
    # CFRP limit is 2000 MPa / 1.5 = 1333 MPa -> margin = 1333/765 = 1.74 > 1.5 (pass)
    if r["stress_margin"] >= 1.5:
        print(f"[OK] Default mission_level_metrics has stress_margin={r['stress_margin']:.2f} (CFRP active)")
    else:
        print(f"[FAIL] Default mission_level_metrics stress_margin={r['stress_margin']:.2f} (likely still BFRP)")

if __name__ == "__main__":
    verify_phase0()
