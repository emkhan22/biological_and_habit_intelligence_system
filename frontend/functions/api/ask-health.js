/**
 * Server-side Gemini 4 Hydrogen Q&A Proxy (`/api/ask-health`).
 * Enforces Gemini 4 API rules: NO temperature, top_p, top_k, or cachedContent;
 * explicit thinkingLevel = "MEDIUM"; optional native google_search grounding.
 * Structurally rejects any smoking-related query in Clean Mode.
 */

const FORBIDDEN_RE = /(smoking|smoke_log|smoking_private|cigarette|smk-)/i;

export async function onRequestPost({ request, env }) {
  try {
    const body = await request.json();
    const question = String(body?.question || "").trim();
    const packet = body?.packet || {};
    const useSearch = body?.use_search !== false;

    if (!question) {
      return new Response(JSON.stringify({ error: "Question is required." }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (FORBIDDEN_RE.test(question)) {
      return new Response(
        JSON.stringify({ error: "AGENTS.md Rule 2: Clean Mode Q&A cannot access private smoking data." }),
        { status: 403, headers: { "Content-Type": "application/json" } }
      );
    }

    const apiKey = env.GEMINI_API_KEY;
    if (!apiKey) {
      return new Response(JSON.stringify({ error: "GEMINI_API_KEY secret not configured." }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }

    const model = "gemini-4-hydrogen";
    const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(
      apiKey
    )}`;

    const payload = {
      systemInstruction: {
        parts: [
          {
            text: "You are the Personal Health Intelligence Advisor. Return strictly valid JSON with keys: what_your_data_shows, what_research_says, citations (array of {title, url}), confidence_tier, epistemic_label (Measured|Descriptive|Correlation|Prediction|Causal_Hypothesis|Screening_Context), ask_your_doctor_note. Never phrase correlations or ML weights as causal proof.",
          },
        ],
      },
      contents: [
        {
          role: "user",
          parts: [
            {
              text: `Question: ${question}\n\nValidated Packet:\n${JSON.stringify(packet).slice(0, 18000)}`,
            },
          ],
        },
      ],
      generationConfig: {
        responseMimeType: "application/json",
        thinkingConfig: { thinkingLevel: "MEDIUM" },
      },
      ...(useSearch ? { tools: [{ googleSearch: {} }] } : {}),
    };

    const gResp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!gResp.ok) {
      throw new Error(`Gemini API returned ${gResp.status}`);
    }
    const gJson = await gResp.json();
    const rawText = gJson?.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
    const parsed = JSON.parse(rawText);
    return new Response(
      JSON.stringify({ ...parsed, engine_badge: `Gemini_AI (${model})` }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  } catch (err) {
    return new Response(JSON.stringify({ error: err?.message || "Q&A proxy failed." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
