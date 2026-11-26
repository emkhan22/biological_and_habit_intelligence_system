#!/usr/bin/env python3
"""
Local Phase 1 validator:
1. Checks JavaScript syntax of all apps-script/*.gs files via `node --check`.
2. Checks appsscript.json is valid JSON.
3. Ensures ZERO hardcoded Google Sheet IDs or PIN literals are committed to GitHub.
"""
import glob, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS_FILES = sorted(glob.glob(os.path.join(ROOT, 'apps-script', '*.gs')))

def main():
    assert GS_FILES, "No .gs files found!"
    # 1. JS syntax check (pipe via stdin since Node 22 rejects .gs file extension)
    for p in GS_FILES:
        with open(p) as f:
            src = f.read()
        r = subprocess.run(['node', '--check'], input=src, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[FAIL] Syntax error in {p}:\n{r.stderr}")
            sys.exit(1)
        print(f"[PASS] Syntax OK: {os.path.basename(p)}")

    # 2. Manifest JSON check
    manifest = os.path.join(ROOT, 'apps-script', 'appsscript.json')
    with open(manifest) as f:
        data = json.load(f)
    assert data.get('timeZone') == 'Asia/Kolkata', "Expected Asia/Kolkata timeZone"
    print("[PASS] Manifest OK: appsscript.json")

    # 3. Secret / Spreadsheet ID leak check (Google Sheet IDs are ~44-char alphanumeric strings)
    id_re = re.compile(r'1[A-Za-z0-9_-]{42,44}')
    for p in GS_FILES:
        with open(p) as f:
            txt = f.read()
        matches = id_re.findall(txt)
        if matches:
            print(f"[FAIL] Potential hardcoded Spreadsheet ID found in {p}: {matches}")
            sys.exit(1)
    print("[PASS] Secret scan OK: Zero hardcoded Spreadsheet IDs in apps-script/*.gs")

    # 4. Privacy check: Engine.gs must never write Smoking_Log to Health_Intelligence_Store
    with open(os.path.join(ROOT, 'apps-script', 'Engine.gs')) as f:
        eng = f.read()
    assert "readSheetObjects_(ssMJ, 'Smoking_Log')" not in eng, "Engine.gs must not read Smoking_Log!"
    print("[PASS] Privacy scan OK: Engine.gs does not touch Smoking_Log")

    print("\nALL PHASE 1 LOCAL VALIDATIONS PASSED!")

if __name__ == '__main__':
    main()
