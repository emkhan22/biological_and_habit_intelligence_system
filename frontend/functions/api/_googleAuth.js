/**
 * Pure Web Standards (Cloudflare Workers / Pages Functions) Google Service Account JWT Auth
 * Uses `crypto.subtle` RS256 signing + `fetch` — ZERO Node-only dependencies.
 */

function base64UrlEncode(input) {
  const bytes = typeof input === "string" ? new TextEncoder().encode(input) : new Uint8Array(input);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function pemToArrayBuffer(pem) {
  const cleaned = pem
    .replace(/-----BEGIN PRIVATE KEY-----/g, "")
    .replace(/-----END PRIVATE KEY-----/g, "")
    .replace(/\s+/g, "");
  const binary = atob(cleaned);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

export async function getGoogleAccessToken(saJsonString, scope = "https://www.googleapis.com/auth/spreadsheets.readonly") {
  if (!saJsonString) {
    throw new Error("Missing Google Service Account JSON secret.");
  }
  const sa = typeof saJsonString === "string" ? JSON.parse(saJsonString) : saJsonString;
  const nowSec = Math.floor(Date.now() / 1000);
  const header = base64UrlEncode(JSON.stringify({ alg: "RS256", typ: "JWT" }));
  const claimSet = base64UrlEncode(
    JSON.stringify({
      iss: sa.client_email,
      scope,
      aud: sa.token_uri || "https://oauth2.googleapis.com/token",
      exp: nowSec + 3600,
      iat: nowSec,
    })
  );
  const signingInput = `${header}.${claimSet}`;

  const key = await crypto.subtle.importKey(
    "pkcs8",
    pemToArrayBuffer(sa.private_key),
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sigBuf = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5",
    key,
    new TextEncoder().encode(signingInput)
  );
  const assertion = `${signingInput}.${base64UrlEncode(sigBuf)}`;

  const tokenResp = await fetch(sa.token_uri || "https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
      assertion,
    }),
  });

  if (!tokenResp.ok) {
    throw new Error(`Google OAuth token exchange failed (${tokenResp.status}).`);
  }
  const tokenJson = await tokenResp.json();
  return tokenJson.access_token;
}

export async function batchReadSheetTabs(accessToken, spreadsheetId, tabNames) {
  const ranges = tabNames.map((t) => `ranges=${encodeURIComponent(`'${t}'!A1:ZZ2000`)}`).join("&");
  const url = `https://sheets.googleapis.com/v4/spreadsheets/${encodeURIComponent(
    spreadsheetId
  )}/values:batchGet?${ranges}&valueRenderOption=UNFORMATTED_VALUE`;

  const resp = await fetch(url, {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (!resp.ok) {
    throw new Error(`Google Sheets batchGet failed (${resp.status}).`);
  }
  const data = await resp.json();
  const tables = {};

  for (let i = 0; i < tabNames.length; i++) {
    const tab = tabNames[i];
    const vr = data.valueRanges?.[i];
    const values = vr?.values || [];
    if (values.length < 2) {
      tables[tab] = [];
      continue;
    }
    const headers = values[0].map((h) => String(h).trim());
    tables[tab] = values.slice(1).map((row) => {
      const obj = {};
      for (let c = 0; c < headers.length; c++) {
        const val = row[c];
        obj[headers[c]] = val === undefined || val === "" ? null : val;
      }
      return obj;
    });
  }
  return tables;
}

export async function sha256Hex(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(String(text)));
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
