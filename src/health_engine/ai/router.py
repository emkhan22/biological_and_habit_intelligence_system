"""Gemini 4 Model Router, API Convention Guardrails, & ₹500/Month Budget Circuit Breaker.

Enforces:
1. Gemini 4 Mandatory API Conventions:
   - Never pass `temperature`, `top_p`, `top_k`, `cached_content`, or `thinking_budget` on Gemini 4
     (`gemini-4-hydrogen`, `gemini-4-neon`, `gemini-4-argon`).
   - Explicit `thinking_level` per task; never pass `minimal` to any Gemini 4 model (and restrict
     `gemini-4-argon` to `low` | `medium` | `high`).
   - HTTP read timeout >= 300s (`timeout_ms = 360000` in `types.HttpOptions`) + exponential backoff
     on transient HTTP `429`/`5xx` and timeout errors.
   - Single-call combination of flat `response_schema` + native `GoogleSearch` grounding when
     `use_google_search=True`, extracting verified `(title, url)` citations from `grounding_metadata`.
2. ₹500/Month Hard Budget Circuit Breaker (`AI_Cost_Ledger`):
   - `>= ₹350` (70%): auto-downgrade `2A_daily_brief` & `4A_interactive_qa_and_simulator` to
     `gemini-3.5-flash-lite`.
   - `>= ₹450` (90%): skip non-essential scheduled jobs (`3C_research_pulse`, `1C_meal_bioavailability`).
   - `>= ₹500` (100%): hard stop all Gemini API calls and return deterministic Python template outputs
     (`engine_badge = "Deterministic_Fallback"`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any, Callable, TypeVar
import uuid

from google import genai
from google.genai import types
from google.genai.errors import APIError
import pandas as pd
from pydantic import BaseModel, ValidationError

from health_engine.config_loader import load_ai_routing_config, now_ist, now_ist_str

T = TypeVar("T", bound=BaseModel)

GEMINI_4_MODELS: frozenset[str] = frozenset(
    {"gemini-4-hydrogen", "gemini-4-neon", "gemini-4-argon"}
)

GEMINI_4_ALLOWED_THINKING: dict[str, frozenset[str]] = {
    "gemini-4-hydrogen": frozenset({"low", "medium", "high", "xhigh"}),
    "gemini-4-neon": frozenset({"low", "medium", "high", "xhigh"}),
    "gemini-4-argon": frozenset({"low", "medium", "high"}),
}

COST_LEDGER_COLUMNS: list[str] = [
    "call_id",
    "timestamp_ist",
    "month_yyyy_mm",
    "module_id",
    "model_used",
    "prompt_tokens",
    "candidates_tokens",
    "thoughts_tokens",
    "grounded_search_queries",
    "estimated_usd",
    "estimated_inr",
    "status",
    "fallback_used",
    "epistemic_class",
]


@dataclass(frozen=True)
class RouteDecision:
    """Pre-call routing & budget circuit breaker decision."""

    module_id: str
    allow_api: bool
    skip_job: bool
    model: str
    thinking_level: str
    use_google_search: bool
    downgraded: bool
    reason: str
    mtd_spend_inr: float


@dataclass
class VerifiedCitation:
    """Verified web citation extracted from Gemini Search `grounding_metadata`."""

    title: str
    url: str


@dataclass
class AICallResult:
    """Structured result from `ModelRouter.generate_structured`."""

    parsed: Any
    engine_badge: str
    model_used: str
    status: str
    fallback_used: bool
    verified_citations: list[VerifiedCitation] = field(default_factory=list)
    web_search_queries: list[str] = field(default_factory=list)
    estimated_inr: float = 0.0
    call_id: str = ""


class GeminiConfigGuard:
    """Runtime validator enforcing Gemini 4 API rules before dispatching requests."""

    @staticmethod
    def is_gemini_4(model: str) -> bool:
        m = str(model or "").strip().lower()
        return m in GEMINI_4_MODELS or m.startswith("gemini-4-")

    @classmethod
    def validate_and_build_config(
        cls,
        *,
        model: str,
        system_instruction: str,
        response_schema: type[BaseModel],
        thinking_level: str | None,
        use_google_search: bool = False,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> types.GenerateContentConfig:
        m = str(model or "").strip()
        extra = dict(extra_kwargs or {})

        if cls.is_gemini_4(m):
            forbidden_keys = {"temperature", "top_p", "top_k", "cached_content", "thinking_budget"}
            violated = forbidden_keys.intersection(
                {k for k, v in extra.items() if v is not None}
            )
            if violated:
                raise ValueError(
                    f"Gemini 4 API violation on '{m}': forbidden parameters {sorted(violated)} passed."
                )
            lvl = str(thinking_level or "").strip().lower()
            if not lvl:
                raise ValueError(f"Explicit thinking_level is required on Gemini 4 model '{m}'.")
            if lvl == "minimal":
                raise ValueError(
                    f"Gemini 4 API violation: thinking_level='minimal' is unsupported on '{m}'."
                )
            allowed = GEMINI_4_ALLOWED_THINKING.get(m, frozenset({"low", "medium", "high"}))
            if lvl not in allowed:
                raise ValueError(
                    f"Invalid thinking_level='{lvl}' for '{m}'. Allowed: {sorted(allowed)}."
                )
            t_cfg = types.ThinkingConfig(thinking_level=lvl)
        else:
            if extra.get("cached_content") is not None:
                raise ValueError("Explicit context caching (cached_content) is disabled by policy.")
            lvl = str(thinking_level or "minimal").strip().lower()
            t_cfg = types.ThinkingConfig(thinking_level=lvl)

        tools_list: list[types.Tool] | None = None
        if use_google_search:
            tools_list = [types.Tool(google_search=types.GoogleSearch())]

        cfg = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=t_cfg,
            tools=tools_list,
        )

        # Final post-construction assertion for Gemini 4
        if cls.is_gemini_4(m):
            assert cfg.temperature is None
            assert cfg.top_p is None
            assert cfg.top_k is None
            assert cfg.cached_content is None
            assert cfg.thinking_config is not None
            assert cfg.thinking_config.thinking_budget is None
        return cfg


class BudgetCircuitBreaker:
    """Evaluates MTD INR spend from `AI_Cost_Ledger` and enforces 70% / 90% / 100% gates."""

    def __init__(self, routing_cfg: dict[str, Any], ledger_df: pd.DataFrame | None = None) -> None:
        self.routing_cfg = routing_cfg
        self.budget_cfg = routing_cfg.get("budget", {})
        self.pricing_cfg = routing_cfg.get("pricing_per_1m_usd", {})
        self.modules_cfg = routing_cfg.get("modules", {})
        self._records: list[dict[str, Any]] = []
        if ledger_df is not None and not ledger_df.empty:
            for row in ledger_df.to_dict(orient="records"):
                self._records.append(dict(row))

    def current_month_key(self) -> str:
        return now_ist().strftime("%Y-%m")

    def month_to_date_spend_inr(self, month_key: str | None = None) -> float:
        mk = month_key or self.current_month_key()
        total = 0.0
        for r in self._records:
            if str(r.get("month_yyyy_mm", "")).strip() == mk:
                try:
                    val = float(r.get("estimated_inr", 0.0) or 0.0)
                    if pd.notna(val):
                        total += val
                except (TypeError, ValueError):
                    continue
        return round(total, 4)

    def month_to_date_search_queries(self, month_key: str | None = None) -> int:
        mk = month_key or self.current_month_key()
        total = 0
        for r in self._records:
            if str(r.get("month_yyyy_mm", "")).strip() == mk:
                try:
                    total += int(float(r.get("grounded_search_queries", 0) or 0))
                except (TypeError, ValueError):
                    continue
        return total

    def estimate_call_cost(
        self,
        *,
        model: str,
        prompt_tokens: int,
        candidates_tokens: int,
        thoughts_tokens: int,
        grounded_search_queries: int,
        month_key: str | None = None,
    ) -> tuple[float, float]:
        """Return `(estimated_usd, estimated_inr)` including output-billed thinking tokens."""
        rates = self.pricing_cfg.get(model, {"input": 1.50, "output": 7.50})
        in_rate = float(rates.get("input", 1.50))
        out_rate = float(rates.get("output", 7.50))
        usd_to_inr = float(self.budget_cfg.get("usd_to_inr", 95.23))

        # Thoughts tokens are billed at output rate by Google Gemini API
        in_usd = (max(0, int(prompt_tokens)) / 1_000_000.0) * in_rate
        out_usd = (max(0, int(candidates_tokens) + int(thoughts_tokens)) / 1_000_000.0) * out_rate

        free_q = int(self.budget_cfg.get("free_monthly_search_queries", 5000))
        per_1k = float(self.budget_cfg.get("search_price_per_1k_usd", 14.0))
        prior_q = self.month_to_date_search_queries(month_key)
        billable_q = max(0, (prior_q + max(0, int(grounded_search_queries))) - max(prior_q, free_q))
        search_usd = (billable_q / 1000.0) * per_1k

        est_usd = round(in_usd + out_usd + search_usd, 6)
        est_inr = round(est_usd * usd_to_inr, 4)
        return est_usd, est_inr

    def evaluate_route(
        self,
        module_id: str,
        *,
        dry_run: bool = False,
        api_key_present: bool = True,
        override_use_search: bool | None = None,
        month_key: str | None = None,
    ) -> RouteDecision:
        mcfg = self.modules_cfg.get(module_id)
        if mcfg is None:
            raise KeyError(f"Unknown AI module_id '{module_id}' in ai_routing.yaml.")

        base_model = str(mcfg["model"])
        base_thinking = str(mcfg.get("thinking_level", "medium"))
        use_search = (
            bool(override_use_search)
            if override_use_search is not None
            else bool(mcfg.get("use_google_search", False))
        )
        essential = bool(mcfg.get("essential", True))

        mtd = self.month_to_date_spend_inr(month_key)
        hard_stop = float(self.budget_cfg.get("hard_stop_threshold_inr", 500.0))
        skip_90 = float(self.budget_cfg.get("skip_non_essential_threshold_inr", 450.0))
        down_70 = float(self.budget_cfg.get("downgrade_threshold_inr", 350.0))

        if dry_run:
            return RouteDecision(
                module_id=module_id,
                allow_api=False,
                skip_job=False,
                model="Deterministic_Python",
                thinking_level="none",
                use_google_search=False,
                downgraded=False,
                reason="Dry_Run",
                mtd_spend_inr=mtd,
            )

        if not api_key_present:
            return RouteDecision(
                module_id=module_id,
                allow_api=False,
                skip_job=False,
                model="Deterministic_Python",
                thinking_level="none",
                use_google_search=False,
                downgraded=False,
                reason="Missing_API_Key",
                mtd_spend_inr=mtd,
            )

        if mtd >= hard_stop:
            return RouteDecision(
                module_id=module_id,
                allow_api=False,
                skip_job=False,
                model="Deterministic_Python",
                thinking_level="none",
                use_google_search=False,
                downgraded=False,
                reason="Budget_Hard_Stop_500_INR",
                mtd_spend_inr=mtd,
            )

        if mtd >= skip_90 and not essential:
            return RouteDecision(
                module_id=module_id,
                allow_api=False,
                skip_job=True,
                model="Skipped_Budget_90Pct",
                thinking_level="none",
                use_google_search=False,
                downgraded=False,
                reason="Skipped_NonEssential_Above_450_INR",
                mtd_spend_inr=mtd,
            )

        if mtd >= down_70 and bool(mcfg.get("downgrade_at_70pct", False)):
            down_model = str(mcfg.get("downgrade_model", "gemini-3.5-flash-lite"))
            down_think = str(mcfg.get("downgrade_thinking_level", "minimal"))
            return RouteDecision(
                module_id=module_id,
                allow_api=True,
                skip_job=False,
                model=down_model,
                thinking_level=down_think,
                use_google_search=use_search,
                downgraded=True,
                reason="Downgraded_Above_350_INR",
                mtd_spend_inr=mtd,
            )

        return RouteDecision(
            module_id=module_id,
            allow_api=True,
            skip_job=False,
            model=base_model,
            thinking_level=base_thinking,
            use_google_search=use_search,
            downgraded=False,
            reason="Primary_Route",
            mtd_spend_inr=mtd,
        )

    def record_call(
        self,
        *,
        module_id: str,
        model_used: str,
        prompt_tokens: int,
        candidates_tokens: int,
        thoughts_tokens: int,
        grounded_search_queries: int,
        status: str,
        fallback_used: bool,
        month_key: str | None = None,
    ) -> dict[str, Any]:
        mk = month_key or self.current_month_key()
        if prompt_tokens == 0 and candidates_tokens == 0 and thoughts_tokens == 0 and grounded_search_queries == 0:
            est_usd, est_inr = 0.0, 0.0
        else:
            est_usd, est_inr = self.estimate_call_cost(
                model=model_used,
                prompt_tokens=prompt_tokens,
                candidates_tokens=candidates_tokens,
                thoughts_tokens=thoughts_tokens,
                grounded_search_queries=grounded_search_queries,
                month_key=mk,
            )
        entry = {
            "call_id": f"AIC-{uuid.uuid4().hex[:12]}",
            "timestamp_ist": now_ist_str(),
            "month_yyyy_mm": mk,
            "module_id": module_id,
            "model_used": model_used,
            "prompt_tokens": int(prompt_tokens),
            "candidates_tokens": int(candidates_tokens),
            "thoughts_tokens": int(thoughts_tokens),
            "grounded_search_queries": int(grounded_search_queries),
            "estimated_usd": est_usd,
            "estimated_inr": est_inr,
            "status": status,
            "fallback_used": bool(fallback_used),
            "epistemic_class": "Measured",
        }
        self._records.append(entry)
        return entry

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=COST_LEDGER_COLUMNS)
        df = pd.DataFrame(self._records)
        for col in COST_LEDGER_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[COST_LEDGER_COLUMNS].copy()


def extract_grounding_citations(response: Any) -> tuple[list[VerifiedCitation], list[str]]:
    """Extract verified `(title, url)` citations and executed search queries from `grounding_metadata`."""
    citations: list[VerifiedCitation] = []
    queries: list[str] = []
    seen_urls: set[str] = set()

    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        meta = getattr(cand, "grounding_metadata", None)
        if meta is None:
            continue
        for q in getattr(meta, "web_search_queries", None) or []:
            qs = str(q).strip()
            if qs and qs not in queries:
                queries.append(qs)
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            uri = str(getattr(web, "uri", "") or "").strip()
            title = str(getattr(web, "title", "") or uri).strip()
            if uri and uri not in seen_urls:
                seen_urls.add(uri)
                citations.append(VerifiedCitation(title=title, url=uri))

    return citations, queries


def _extract_usage_counts(response: Any) -> tuple[int, int, int]:
    """Return `(prompt_plus_tool_tokens, candidates_tokens, thoughts_tokens)` from response."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0, 0
    prompt_t = int(getattr(usage, "prompt_token_count", 0) or 0)
    tool_prompt_t = int(getattr(usage, "tool_use_prompt_token_count", 0) or 0)
    cand_t = int(getattr(usage, "candidates_token_count", 0) or 0)
    thought_t = int(getattr(usage, "thoughts_token_count", 0) or 0)
    return prompt_t + tool_prompt_t, cand_t, thought_t


class ModelRouter:
    """Centralized `google-genai` execution wrapper with retries, grounding, and deterministic fallbacks."""

    def __init__(
        self,
        *,
        routing_cfg: dict[str, Any] | None = None,
        ledger_df: pd.DataFrame | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        dry_run: bool = False,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.routing_cfg = routing_cfg or load_ai_routing_config()
        self.breaker = BudgetCircuitBreaker(self.routing_cfg, ledger_df=ledger_df)
        self.dry_run = dry_run
        self._sleep = sleep_fn
        self._injected_client = client

        resolved_key = (
            api_key
            if api_key is not None
            else os.environ.get("GEMINI_API_KEY", "").strip()
        )
        self.api_key = resolved_key
        self._client: Any | None = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        timeout_ms = int(self.routing_cfg.get("http", {}).get("timeout_ms", 360_000))
        if timeout_ms < 300_000:
            timeout_ms = 300_000
        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        return self._client

    def _invoke_with_backoff(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig,
    ) -> Any:
        http_cfg = self.routing_cfg.get("http", {})
        max_retries = max(1, int(http_cfg.get("max_api_retries", 3)))
        base_delay = float(http_cfg.get("base_backoff_sec", 2.0))
        client = self._get_client()

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except APIError as exc:
                last_exc = exc
                code = getattr(exc, "code", None)
                if code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                    self._sleep(min(60.0, base_delay * (2**attempt)))
                    continue
                raise
            except (TimeoutError, ConnectionError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    self._sleep(min(60.0, base_delay * (2**attempt)))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unreachable retry loop exit.")

    def generate_structured(
        self,
        *,
        module_id: str,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type[T],
        fallback_fn: Callable[[], T],
        override_use_search: bool | None = None,
        extra_config_kwargs: dict[str, Any] | None = None,
        month_key: str | None = None,
    ) -> AICallResult:
        """Execute routed Gemini call with schema validation, 1 retry, and deterministic fallback."""
        has_key = bool(self.api_key) or (self._injected_client is not None)
        decision = self.breaker.evaluate_route(
            module_id,
            dry_run=self.dry_run,
            api_key_present=has_key,
            override_use_search=override_use_search,
            month_key=month_key,
        )

        if decision.skip_job:
            rec = self.breaker.record_call(
                module_id=module_id,
                model_used=decision.model,
                prompt_tokens=0,
                candidates_tokens=0,
                thoughts_tokens=0,
                grounded_search_queries=0,
                status=decision.reason,
                fallback_used=True,
                month_key=month_key,
            )
            return AICallResult(
                parsed=None,
                engine_badge="Skipped_Budget_90Pct",
                model_used=decision.model,
                status=decision.reason,
                fallback_used=True,
                call_id=str(rec["call_id"]),
            )

        if not decision.allow_api:
            fb_obj = fallback_fn()
            rec = self.breaker.record_call(
                module_id=module_id,
                model_used=decision.model,
                prompt_tokens=0,
                candidates_tokens=0,
                thoughts_tokens=0,
                grounded_search_queries=0,
                status=decision.reason,
                fallback_used=True,
                month_key=month_key,
            )
            return AICallResult(
                parsed=fb_obj,
                engine_badge="Deterministic_Fallback",
                model_used=decision.model,
                status=decision.reason,
                fallback_used=True,
                call_id=str(rec["call_id"]),
            )

        # Validate Gemini 4 API guardrails strictly BEFORE network call (raises ValueError on violation)
        config = GeminiConfigGuard.validate_and_build_config(
            model=decision.model,
            system_instruction=system_prompt,
            response_schema=schema_cls,
            thinking_level=decision.thinking_level,
            use_google_search=decision.use_google_search,
            extra_kwargs=extra_config_kwargs,
        )

        schema_retries = int(self.routing_cfg.get("http", {}).get("schema_validation_retries", 1))
        total_prompt_t = 0
        total_cand_t = 0
        total_thought_t = 0
        all_citations: list[VerifiedCitation] = []
        all_queries: list[str] = []

        current_prompt = user_prompt
        last_status = "API_Error"

        for val_attempt in range(schema_retries + 1):
            try:
                resp = self._invoke_with_backoff(
                    model=decision.model,
                    contents=current_prompt,
                    config=config,
                )
                p_t, c_t, th_t = _extract_usage_counts(resp)
                total_prompt_t += p_t
                total_cand_t += c_t
                total_thought_t += th_t

                cites, qs = extract_grounding_citations(resp)
                for c in cites:
                    if all(existing.url != c.url for existing in all_citations):
                        all_citations.append(c)
                for q in qs:
                    if q not in all_queries:
                        all_queries.append(q)

                raw_text = str(getattr(resp, "text", "") or "").strip()
                if not raw_text:
                    raise ValueError("Empty response.text returned by Gemini model.")

                parsed_obj = schema_cls.model_validate_json(raw_text)
                status_str = "Success_Downgraded" if decision.downgraded else "Success"
                rec = self.breaker.record_call(
                    module_id=module_id,
                    model_used=decision.model,
                    prompt_tokens=total_prompt_t,
                    candidates_tokens=total_cand_t,
                    thoughts_tokens=total_thought_t,
                    grounded_search_queries=len(all_queries),
                    status=status_str,
                    fallback_used=False,
                    month_key=month_key,
                )
                badge = f"Gemini_AI ({decision.model})"
                return AICallResult(
                    parsed=parsed_obj,
                    engine_badge=badge,
                    model_used=decision.model,
                    status=status_str,
                    fallback_used=False,
                    verified_citations=all_citations,
                    web_search_queries=all_queries,
                    estimated_inr=float(rec["estimated_inr"]),
                    call_id=str(rec["call_id"]),
                )
            except (ValidationError, ValueError) as val_exc:
                last_status = f"Schema_Validation_Failed_Attempt_{val_attempt + 1}"
                if val_attempt < schema_retries:
                    current_prompt = (
                        f"{user_prompt}\n\n[VALIDATION CORRECTION REQUIRED]: Your prior JSON output failed "
                        f"strict schema validation ({val_exc}). Return strictly valid JSON adhering to the schema "
                        f"and valid epistemic_label values without causal overclaims."
                    )
                    continue
            except Exception as api_exc:
                last_status = f"API_Fallback ({type(api_exc).__name__})"
                break

        # Graceful deterministic Python fallback after validation retry or API/quota failure
        fb_obj = fallback_fn()
        rec = self.breaker.record_call(
            module_id=module_id,
            model_used=decision.model,
            prompt_tokens=total_prompt_t,
            candidates_tokens=total_cand_t,
            thoughts_tokens=total_thought_t,
            grounded_search_queries=len(all_queries),
            status=f"Fallback_After_{last_status}",
            fallback_used=True,
            month_key=month_key,
        )
        return AICallResult(
            parsed=fb_obj,
            engine_badge="Deterministic_Fallback",
            model_used=decision.model,
            status=str(rec["status"]),
            fallback_used=True,
            verified_citations=all_citations,
            web_search_queries=all_queries,
            estimated_inr=float(rec["estimated_inr"]),
            call_id=str(rec["call_id"]),
        )
