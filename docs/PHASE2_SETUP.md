# Phase 2 Setup Guide — Deploying the 4-Tab Glapp+ Mobile Web App & Google Sites Embed

This deploys your private, **\$0, zero-API-key** mobile-first Web App on top of your working Phase 1 engine (~5 minutes).

---

## Step 1: Add the Phase 2 Files in Apps Script
1. Open **`Health_Intelligence_Store`** → **Extensions → Apps Script**.
2. Update **`appsscript.json`** with `apps-script/appsscript.json` (adds the `MYSELF` / `USER_DEPLOYING` Web App security lock).
3. Add **1 new Script file**:
   - `WebApp.gs` (paste `apps-script/WebApp.gs`)
4. Add **4 new HTML files** (`+` → **HTML**, exact casing matters!):
   - `Index` (paste `apps-script/Index.html`)
   - `Styles` (paste `apps-script/Styles.html`)
   - `Charts` (paste `apps-script/Charts.html`)
   - `App` (paste `apps-script/App.html`)
5. Click **Save** (`Ctrl+S` / `Cmd+S`).

---

## Step 2: Deploy Privately as Your Personal Web App
1. Click **Deploy → New deployment**.
2. Click the gear icon → choose **Web app**.
3. Configure **strictly** as:
   - **Description:** `Glapp+ Health Intelligence V1`
   - **Execute as:** `Me` *(your Google account)*
   - **Who has access:** `Only myself` *(NEVER choose "Anyone"!)*
4. Click **Deploy** → copy the **Web app URL** (ends in `/exec`).

---

## Step 3: Open on Mobile & Embed in Your Private Google Site
### On Your Phone (Glapp-like App Feel)
- Open the `/exec` URL in Chrome/Safari while signed into your Google account → tap **Add to Home Screen** for a 1-tap dark-mode mobile experience.

### Inside Google Sites (Desktop + Mobile Portal)
1. Create or open your private **Google Site** (Share settings: restricted to **only your Google account**).
2. Click **Insert → Embed → By URL** → paste your `/exec` Web app URL → choose **Whole page** → resize full-width and **Publish**.

---

## Step 4: Quick Phase 2 Verification Checklist
- [ ] **Clean Mode (default):** Top-right button says `🔒 Clean`, Weekly Readiness Ring shows **5 pillars**, and `Cravings` chip is hidden.
- [ ] **Private Mode (PIN 2):** Tap `🔒 Clean` → enter **PIN 2** → button turns rose `🔓 Private`, ring expands to **6 pillars**, and `🔒 Cravings` unlocks in Progress.
- [ ] **Re-Lock (PIN 1 or Lock button):** Immediately returns to Clean Mode and clears private session token.
- [ ] **Drill-Downs & Simulators:** Tap `Body Comp`, `Symptoms`, `Sleep`, and `Fatty Liver` cards; toggle `W | M | 3M | All` and Multi-Event Overlays.
- [ ] **Reports & Print:** Switch `Week 1` → latest in `Reports`, load the **Whole-Body Doctor Visit Brief**, and test **Print / Save PDF**.

Ready for **Phase 3 (Google AI Pro Daily, Friday Weekly, Monthly & Wisdom Gem prompts)**!
