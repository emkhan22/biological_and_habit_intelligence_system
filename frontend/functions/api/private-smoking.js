import { batchReadSheetTabs, getGoogleAccessToken, sha256Hex } from "./_googleAuth.js";

/**
 * Strictly Isolated Private Smoking Endpoint (`/api/private-smoking`).
 *
 * Security Guarantees (AGENTS.md Rule 2):
 * 1. Verifies Cloudflare Access email (`Cf-Access-Authenticated-User-Email`) against
 *    `PRIVATE_ALLOWED_EMAILS` (with explicit local/dev fallback ONLY when `ALLOW_LOCAL_DEV_AUTH="true"`).
 * 2. Verifies secondary PIN against SHA-256 hash in `PRIVATE_PIN_SHA256`.
 * 3. Reads ONLY `Smoking_Private_Store` (`SHEET_ID_SMOKING_STORE`) via `GCP_SMOKING_SA_JSON`.
 * 4. Refuses execution if `SHEET_ID_SMOKING_STORE === SHEET_ID_STORE_V2`.
 */

const PRIVATE_TABS = [
  "Private_Smoking_Descriptive",
  "Private_Smoking_Risk_Drivers",
  "Private_Delay_Outcome_Stats",
  "Private_AI_Coach",
  "Private_Model_Audit",
];

export async function onRequestPost({ request, env }) {
  try {
    const body = await request.json();
    const suppliedEmail = String(body?.email || "").trim().toLowerCase();
    const suppliedPin = String(body?.pin || "").trim();

    const cfEmail = String(request.headers.get("Cf-Access-Authenticated-User-Email") || "")
      .trim()
      .toLowerCase();
    const allowLocalDev = String(env.ALLOW_LOCAL_DEV_AUTH || "").toLowerCase() === "true";

    const verifiedEmail = cfEmail || (allowLocalDev ? suppliedEmail : "");
    if (!verifiedEmail) {
      return new Response(
        JSON.stringify({
          error:
            "Missing Cloudflare Access authenticated email header (set ALLOW_LOCAL_DEV_AUTH=true for local wrangler dev).",
        }),
        { status: 401, headers: { "Content-Type": "application/json" } }
      );
    }

    const allowlist = String(env.PRIVATE_ALLOWED_EMAILS || "")
      .split(",")
      .map((e) => e.trim().toLowerCase())
      .filter(Boolean);

    if (allowlist.length > 0 && !allowlist.includes(verifiedEmail)) {
      return new Response(JSON.stringify({ error: "Email not authorized for Private Mode." }), {
        status: 403,
        headers: { "Content-Type": "application/json" },
      });
    }

    const expectedPinHash = String(env.PRIVATE_PIN_SHA256 || "").trim().toLowerCase();
    if (expectedPinHash) {
      const actualHash = await sha256Hex(suppliedPin);
      if (actualHash !== expectedPinHash) {
        return new Response(JSON.stringify({ error: "Invalid secondary PIN." }), {
          status: 403,
          headers: { "Content-Type": "application/json" },
        });
      }
    } else if (!allowLocalDev) {
      return new Response(JSON.stringify({ error: "PRIVATE_PIN_SHA256 secret not configured." }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }

    const smokingSa = env.GCP_SMOKING_SA_JSON;
    const smokingStoreId = String(env.SHEET_ID_SMOKING_STORE || "").trim();
    const publicStoreId = String(env.SHEET_ID_STORE_V2 || "").trim();

    if (smokingStoreId && publicStoreId && smokingStoreId === publicStoreId) {
      return new Response(
        JSON.stringify({ error: "Isolation violation: SHEET_ID_SMOKING_STORE cannot match SHEET_ID_STORE_V2." }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    if (!smokingSa || !smokingStoreId) {
      if (allowLocalDev) {
        return new Response(
          JSON.stringify({
            source: "Smoking_Private_Store (Local Dev Stub)",
            unlocked_email: verifiedEmail,
            tables: {
              Private_Smoking_Descriptive: [],
              Private_Smoking_Risk_Drivers: [],
              Private_Delay_Outcome_Stats: [],
              Private_AI_Coach: [],
              Private_Model_Audit: [],
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } }
        );
      }
      return new Response(JSON.stringify({ error: "Smoking Private Store secrets not configured." }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }

    const token = await getGoogleAccessToken(smokingSa);
    const tables = await batchReadSheetTabs(token, smokingStoreId, PRIVATE_TABS);

    return new Response(
      JSON.stringify({
        source: "Smoking_Private_Store (Live)",
        unlocked_email: verifiedEmail,
        tables,
      }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "no-store",
        },
      }
    );
  } catch (err) {
    return new Response(JSON.stringify({ error: err?.message || "Private Mode fetch failed." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
