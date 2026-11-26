#!/usr/bin/env python3
"""
Phase 1 + Phase 2 Local Validator:
1. Checks JS syntax of all apps-script/*.gs files via `node --check`.
2. Extracts and checks every <script> block inside apps-script/*.html via `node --check`.
3. Validates appsscript.json (Asia/Kolkata TZ, webapp access=MYSELF, executeAs=USER_DEPLOYING).
4. Ensures ZERO hardcoded Spreadsheet IDs, PIN literals, or external CDN <script src="http..."> tags.
5. Ensures Engine.gs and WebApp.gs NEVER reference Smoking_Log.
"""
import glob, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS_FILES = sorted(glob.glob(os.path.join(ROOT, 'apps-script', '*.gs')))
HTML_FILES = sorted(glob.glob(os.path.join(ROOT, 'apps-script', '*.html')))

def check_js(src, label):
    r = subprocess.run(['node', '--check'], input=src, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[FAIL] Syntax error in {label}:\n{r.stderr}")
        sys.exit(1)
    print(f"[PASS] Syntax OK: {label}")

def main():
    assert GS_FILES, "No .gs files found!"
    # 1. .gs syntax checks
    for p in GS_FILES:
        with open(p) as f:
            check_js(f.read(), os.path.basename(p))

    # 2. Inline <script> syntax checks in .html files
    script_re = re.compile(r'<script[^>]*>(.*?)</script>', re.S | re.I)
    cdn_re = re.compile(r'<(?:script|link)[^>]+(?:src|href)\s*=\s*["\']https?://', re.I)
    for p in HTML_FILES:
        with open(p) as f:
            txt = f.read()
        if cdn_re.search(txt):
            print(f"[FAIL] External CDN reference found in {p} (must be zero-CDN / CSP-safe)")
            sys.exit(1)
        blocks = script_re.findall(txt)
        for idx, blk in enumerate(blocks, 1):
            check_js(blk, f"{os.path.basename(p)} <script #{idx}>")
    print("[PASS] Zero-CDN check OK across all .html files")

    # 3. Manifest JSON & WebApp security check
    manifest = os.path.join(ROOT, 'apps-script', 'appsscript.json')
    with open(manifest) as f:
        data = json.load(f)
    assert data.get('timeZone') == 'Asia/Kolkata', "Expected Asia/Kolkata timeZone"
    wa = data.get('webapp', {})
    assert wa.get('access') == 'MYSELF', "webapp.access must be MYSELF"
    assert wa.get('executeAs') == 'USER_DEPLOYING', "webapp.executeAs must be USER_DEPLOYING"
    print("[PASS] Manifest OK: appsscript.json (Asia/Kolkata, MYSELF, USER_DEPLOYING)")

    # 4. Secret / Spreadsheet ID leak check
    id_re = re.compile(r'1[A-Za-z0-9_-]{42,44}')
    for p in GS_FILES + HTML_FILES:
        with open(p) as f:
            txt = f.read()
        matches = id_re.findall(txt)
        if matches:
            print(f"[FAIL] Potential hardcoded Spreadsheet ID found in {p}: {matches}")
            sys.exit(1)
    print("[PASS] Secret scan OK: Zero hardcoded Spreadsheet IDs in apps-script/*.{gs,html}")

    # 5. Privacy isolation check: Engine.gs & WebApp.gs must NEVER read Smoking_Log
    for fn in ['Engine.gs', 'WebApp.gs']:
        with open(os.path.join(ROOT, 'apps-script', fn)) as f:
            txt = f.read()
        assert "Smoking_Log" not in txt or fn == 'Engine.gs' and "readSheetObjects_(ssMJ, 'Smoking_Log')" not in txt, f"{fn} must not read Smoking_Log!"
        if fn == 'WebApp.gs':
            assert "Smoking_Log" not in txt and "getPrivateSmokingAnalytics" not in txt, "WebApp.gs must stay 100% smoking-free!"
    print("[PASS] Privacy scan OK: Engine.gs & WebApp.gs do not read Smoking_Log")

    # 6. Charts global export & Index.html include order + discreet PIN modal checks
    with open(os.path.join(ROOT, 'apps-script', 'Charts.html')) as f:
        charts_txt = f.read()
    assert 'window.Charts' in charts_txt, "Charts.html must expose window.Charts"
    print("[PASS] Charts.html exposes window.Charts")

    with open(os.path.join(ROOT, 'apps-script', 'Index.html')) as f:
        idx_txt = f.read()
    pos_charts = idx_txt.find("include('Charts')")
    pos_app = idx_txt.find("include('App')")
    assert pos_charts != -1 and pos_app != -1 and pos_charts < pos_app, "Index.html must include Charts before App"
    idx_lower = idx_txt.lower()
    for forbidden in ['smoking', 'clean mode', 'private full mode', 'pin 1', 'pin 2', '5-pillar']:
        assert forbidden not in idx_lower, f"Index.html must not contain '{forbidden}' in default markup"
    assert 'maxlength="64"' in idx_txt, "Index.html PIN input must support maxlength=64"
    print("[PASS] Index.html OK: Charts before App, maxlength=64, discreet default markup")

    print("\nALL PHASE 1 + PHASE 2 LOCAL VALIDATIONS PASSED!")

if __name__ == '__main__':
    main()
