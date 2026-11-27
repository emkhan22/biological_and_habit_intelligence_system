# Phase 1 Setup Guide — Data, Math, QA & Privacy Engine

This sets up your **\$0, zero-API-key** daily backend in Google Apps Script (~5 minutes).

---

## Step 1: Open Apps Script Inside `Health_Intelligence_Store`
1. Open your **`Health_Intelligence_Store`** Google Sheet in `Aggregated/`.
2. Go to **Extensions → Apps Script**.
3. In the left sidebar, click **Project Settings (gear icon)** → check **"Show `appsscript.json` manifest file in editor"**.

## Step 2: Copy the 6 Phase 1 Files
*(Or use optional `clasp push` if you prefer CLI syncing.)*
1. Replace `appsscript.json` with `apps-script/appsscript.json`.
2. Create/paste these 5 script files:
   - `Config.gs`
   - `Engine.gs`
   - `DoctorBrief.gs`
   - `PrivacyAuth.gs`
   - `Tests.gs`
3. Click **Save** (`Ctrl+S` / `Cmd+S`).

## Step 3: One-Time Configuration (Stores IDs & PIN Hashes Securely on Google's Server)
In a temporary scratch function (or directly in the Apps Script editor), run these **two setup calls once**, then delete your scratch function:

```javascript
function oneTimeSetup() {
  setupConfig({
    MEDICAL: '<AggregatedMedicalRecords_Spreadsheet_ID>',
    GLAPP:   '<Glapp_Data_Spreadsheet_ID>',
    HC:      '<HealthConnectDaily_Spreadsheet_ID>',
    JOURNAL: '<Micro_Journal_Spreadsheet_ID>',
    STORE:   '<Health_Intelligence_Store_Spreadsheet_ID>'
  });
  setupPrivatePins('YOUR_CLEAN_PIN_1', 'YOUR_PRIVATE_PIN_2');
}
```
*(When Google asks for permissions on first run, approve access for your own account.)*

## Step 4: Run Engine & Automated Self-Test
1. Select **`runDailyHealthEngine`** in the top toolbar → click **Run**.
2. Select **`runPhase1SelfTest`** → click **Run** → verify **Execution Log** prints `ALL 6 PHASE 1 SELF-TESTS PASSED!`.

## Step 5: Schedule Daily 7:00 AM IST Trigger
1. Click **Triggers (clock icon)** in the left sidebar → **+ Add Trigger**.
2. Choose:
   - Function: **`runDailyHealthEngine`**
   - Event source: **Time-driven**
   - Type: **Day timer**
   - Time of day: **7am to 8am** *(runs 2 hours after your 5:00 AM IST upstream ETL)*
3. Click **Save**.

Your `Health_Intelligence_Store` now refreshes automatically every morning! Ready for **Phase 2 (4-Tab Glapp+ Mobile Web App)**.
