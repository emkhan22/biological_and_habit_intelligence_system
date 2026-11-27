/**
 * Server-side Simulator Advisor Proxy (`/api/simulator-advice`).
 * Uses Gemini 4 Hydrogen (`thinkingLevel: "MEDIUM"`, no temperature/top_p/top_k).
 */

export async function onRequestPost({ request, env }) {
  try {
    const body = await request.json();
    const targetKg = Number(body?.target_weight_kg);
    const sliders = body?.sliders_dict || {};
    const packet = body?.packet || {};

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
            text: "You are a Metabolic & Lean-Mass Simulator Advisor. Return strictly valid JSON with keys: what_your_data_shows, what_research_says, realistic_timeline_and_lean_mass_guardrail, prioritized_lever_adjustments (array of strings), citations (array of {title, url}), confidence_tier, epistemic_label ('Prediction'), ask_your_doctor_note.",
          },
        ],
      },
      contents: [
        {
          role: "user",
          parts: [
            {
              text: `Target Weight: ${targetKg} kg\nSliders: ${JSON.stringify(sliders)}\nPacket:\n${JSON.stringify(
                packet
              ).slice(0, 14000)}`,
            },
          ],
        },
      ],
      generationConfig: {
        responseMimeType: "application/json",
        thinkingConfig: { thinkingLevel: "MEDIUM" },
      },
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
    return new Response(JSON.stringify({ error: err?.message || "Simulator advisor failed." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
