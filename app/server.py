"""FastAPI app: paste medical text, get a plain-language rewrite.

The UI shows the fine-tuned (adapter) rewrite alongside the base-model rewrite
and the Flesch-Kincaid reading grade level of each, so the learned capability
is visible directly in the deployed app.

The model loads once at startup. On free CPU hosting the first request is slow
(the model is large for CPU); subsequent requests are faster.
"""

from pathlib import Path
from typing import Dict

import textstat
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from model import Rewriter  # noqa: E402

app = FastAPI(title="PlainMed")

# Load both once. use_adapter=True merges the LoRA weights; False is base only.
_adapter_rewriter: Rewriter | None = None
_base_rewriter: Rewriter | None = None


def get_adapter() -> Rewriter:
    """Lazily load the fine-tuned rewriter."""
    global _adapter_rewriter
    if _adapter_rewriter is None:
        _adapter_rewriter = Rewriter(use_adapter=True)
    return _adapter_rewriter


def get_base() -> Rewriter:
    """Lazily load the base rewriter (for the before/after comparison)."""
    global _base_rewriter
    if _base_rewriter is None:
        _base_rewriter = Rewriter(use_adapter=False)
    return _base_rewriter


class RewriteRequest(BaseModel):
    """Request body for the rewrite endpoint."""

    text: str
    compare: bool = True


def _grade(text: str) -> float:
    """Flesch-Kincaid grade level, guarded against empty strings."""
    return round(textstat.flesch_kincaid_grade(text), 1) if text.strip() else 0.0


@app.post("/api/rewrite")
def rewrite(req: RewriteRequest) -> Dict:
    """Return the plain-language rewrite(s) and their reading grade levels."""
    result: Dict = {
        "source": req.text,
        "source_grade": _grade(req.text),
        "plain": get_adapter().rewrite(req.text),
    }
    result["plain_grade"] = _grade(result["plain"])
    if req.compare:
        result["base"] = get_base().rewrite(req.text)
        result["base_grade"] = _grade(result["base"])
    return result


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-page UI."""
    return INDEX_HTML


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PlainMed — plain-language medical rewriter</title>
<style>
  :root { --ink:#12203a; --sub:#5a6b86; --line:#dfe6f0; --accent:#2E7DA3;
          --bg:#f6f8fb; --card:#fff; --good:#1f7a54; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,
         sans-serif; background:var(--bg); color:var(--ink); line-height:1.5; }
  .wrap { max-width:820px; margin:0 auto; padding:40px 20px 80px; }
  h1 { font-size:1.7rem; margin:0 0 4px; letter-spacing:-0.02em; }
  p.sub { color:var(--sub); margin:0 0 28px; }
  textarea { width:100%; min-height:120px; padding:14px; border:1px solid
             var(--line); border-radius:12px; font-size:1rem; resize:vertical;
             font-family:inherit; background:var(--card); }
  button { margin-top:14px; background:var(--accent); color:#fff; border:0;
           padding:12px 22px; border-radius:10px; font-size:1rem; cursor:pointer;
           font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:12px; padding:18px 20px; margin-top:18px; }
  .label { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em;
           color:var(--sub); margin-bottom:6px; font-weight:700; }
  .grade { float:right; font-weight:700; color:var(--accent); }
  .plain .grade { color:var(--good); }
  .foot { margin-top:32px; font-size:.82rem; color:var(--sub); }
  .disclaimer { background:#fff6ed; border:1px solid #f2d3ad; color:#8a5a1f;
                padding:12px 14px; border-radius:10px; font-size:.85rem;
                margin-top:18px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>PlainMed</h1>
  <p class="sub">Rewrites clinical text into plain language. Fine-tuned
     Qwen2.5-1.5B (LoRA). Shows reading grade level before and after.</p>

  <textarea id="input" placeholder="Paste a sentence from a clinical note, "
    "discharge summary, or research abstract..."></textarea>
  <button id="go" onclick="run()">Rewrite</button>

  <div id="out"></div>

  <div class="disclaimer">Educational demo only. Not medical advice. Outputs
     may be inaccurate and must not be used for clinical decisions.</div>
  <p class="foot">Grade level is Flesch-Kincaid — it measures word and sentence
     length, not medical correctness.</p>
</div>

<script>
async function run() {
  const btn = document.getElementById('go');
  const text = document.getElementById('input').value.trim();
  if (!text) return;
  btn.disabled = true; btn.textContent = 'Rewriting...';
  document.getElementById('out').innerHTML = '';
  try {
    const res = await fetch('/api/rewrite', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ text, compare: true })
    });
    const d = await res.json();
    document.getElementById('out').innerHTML = `
      <div class="card">
        <div class="label">Original <span class="grade">grade ${d.source_grade}</span></div>
        ${escapeHtml(d.source)}
      </div>
      <div class="card plain">
        <div class="label">Plain language — fine-tuned <span class="grade">grade ${d.plain_grade}</span></div>
        ${escapeHtml(d.plain)}
      </div>
      <div class="card">
        <div class="label">Base model (before fine-tuning) <span class="grade">grade ${d.base_grade}</span></div>
        ${escapeHtml(d.base)}
      </div>`;
  } catch (e) {
    document.getElementById('out').innerHTML =
      '<div class="card">Something went wrong. Try again.</div>';
  } finally {
    btn.disabled = false; btn.textContent = 'Rewrite';
  }
}
function escapeHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
</script>
</body>
</html>
"""
