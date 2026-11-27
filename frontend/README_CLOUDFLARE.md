# Cloudflare Pages + Zero Trust Access Deployment Guide (V3.0 PWA)

This guide covers deploying `frontend/` to **Cloudflare Pages** with pure Web Standards Pages Functions (`frontend/functions/api/`), protecting the dashboard with **Cloudflare Access (One-Time PIN)**, and configuring Clean Mode vs. isolated Private Mode secrets.

---

## 1. Cloudflare Pages Git Deployment
1. Go to **Cloudflare Dashboard → Workers & Pages → Create → Pages → Connect to Git**.
2. Select repository: `emkhan22/biological_and_habit_intelligence_system`.
3. Configure build settings:
   - **Production branch:** `main`
   - **Root directory:** `frontend`
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
4. Cloudflare automatically detects `frontend/functions/api/*.js` and deploys them at:
   - `/api/health-data` (Clean Mode reader for **`Health Intelligence Store V2`**)
   - `/api/ask-health` (Server-side Gemini 4 Hydrogen Q&A proxy)
   - `/api/simulator-advice` (Server-side Lean-Mass Simulator Advisor)
   - `/api/private-smoking` (Air-gapped Private Mode endpoint for `Smoking_Private_Store`)

---

## 2. Environment Variables & Secrets (Pages → Settings → Variables and Secrets)

Add the following **Encrypted Secrets** for Production (and Preview if desired):

| Variable | Scope | Purpose |
| :--- | :--- | :--- |
| `HEALTH_READER_SA_JSON` | Clean Mode | Read-only Google Service Account JSON with Viewer access ONLY to `Health_Intelligence_Store_v2`. |
| `SHEET_ID_STORE_V2` | Clean Mode | Spreadsheet ID of **`Health_Intelligence_Store_v2`** (`13nP7Ido1YEMwT_HNBi14QiO4SrMotdrSCCJdR12zn00`). |
| `GEMINI_API_KEY` | Clean Mode | Paid Tier 1 Gemini API key (stays strictly server-side; never exposed in browser bundle). |
| `GCP_SMOKING_SA_JSON` | Private Mode | Isolated Service Account JSON with access ONLY to `Smoking_Private_Store`. |
| `SHEET_ID_SMOKING_STORE` | Private Mode | Dedicated `Smoking_Private_Store` Spreadsheet ID (verified distinct from `SHEET_ID_STORE_V2`). |
| `PRIVATE_ALLOWED_EMAILS` | Private Mode | Comma-separated 1–2 email allowlist permitted to unlock `Cravings` view. |
| `PRIVATE_PIN_SHA256` | Private Mode | Lowercase hex SHA-256 hash of your secondary PIN (`printf 'YOUR_PIN' \| shasum -a 256`). |
| `ALLOW_LOCAL_DEV_AUTH` | Local Dev Only | Set to `true` ONLY in local `.dev.vars` when testing with `wrangler pages dev` without Cloudflare Access headers. **Never enable in Production.** |

---

## 3. Cloudflare Zero Trust Access (One-Time PIN + `*` Wildcard Preview Fix)

1. Go to **Cloudflare Zero Trust → Access → Applications → Add an application → Self-hosted**.
2. **Application Domain & Wildcard Subdomain Fix:**
   - Add your production domain (e.g., `health-intel.pages.dev` or your custom domain).
   - **Important (`*` Wildcard Fix):** To ensure Cloudflare Pages preview deployments (`<hash>.health-intel.pages.dev`) are ALSO protected by Access instead of leaking publicly, add a second hostname with subdomain `*` on `health-intel.pages.dev` (i.e., `*.health-intel.pages.dev`).
3. **Authentication Policy (2-Email One-Time PIN Allowlist):**
   - Policy action: **Allow**
   - Include rule → **Emails**: add your 2 authorized email addresses (e.g., primary + backup).
   - Identity provider: **One-time PIN** (zero password management required).
4. Once enabled, Cloudflare injects `Cf-Access-Authenticated-User-Email` on every request, which `/api/private-smoking` verifies before checking your secondary PIN hash.

---

## 4. Static Hosting (`dist/`) & Offline Preview Notes

- If you host `frontend/dist/` on any static host (GitHub Pages, Netlify static, S3, or `npm run preview`) without Cloudflare Pages Functions:
  - The PWA automatically falls back to `/mock_data.json` (exported from validated Clean Mode tables via `python3 scripts/export_frontend_mock.py`).
  - Interactive Q&A and Simulator Advisor degrade gracefully to deterministic offline responses without requiring `GEMINI_API_KEY`.
  - Private Mode remains locked because `/api/private-smoking` is not exposed on static hosts.
