/* The LLM log: what the planner was asked, what it answered, and what that cost.
 *
 * A routing experiment can fail silently. If the API key is wrong, or a response fails
 * schema validation, or the model names an unavailable subsystem, the orchestrator degrades
 * to the scripted decision and the run completes looking entirely normal — with the arm
 * under test having made none of the calls. The fallback count is the one number that
 * distinguishes that run from a real one.
 *
 * That split is why this file exports two things rather than one. The *summary* stays in the
 * run view's right column, because the fallback warning has to be visible without being
 * looked for. The *log itself* opens in a drawer, because it is a reading surface: a prompt
 * is several thousand tokens, and the column is 300px wide — which is how this panel came to
 * be unreadable despite containing everything it should. Reading material gets width and its
 * own scroll; a warning light gets a fixed place you don't have to scroll to.
 */

import { useEffect, useRef, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { getLlmLog } from '@/api.js';

/* --- the compact card, in the run column ---------------------------------- */

export function LlmLogSummary({ runId, nEvents, onOpen }) {
  const { data, err } = useLlmLog(runId, nEvents, false);

  if (err) return card(html`<div class="small" style="color:var(--err)">${err}</div>`);
  if (!data) return card(html`<div class="empty">loading…</div>`);

  if (!data.uses_llm) {
    return card(html`
      <div class="small muted">
        This run used the <b>${data.policy || 'rule-based'}</b> policy, which makes no API
        calls. Every decision came from code in the repository.
        ${data.n_fallback_decisions > 0 &&
        html`<div style="margin-top:6px;color:var(--warn)">
          ${data.n_fallback_decisions} decision(s) still fell back, meaning the policy itself
          raised. Check the trace for <code class="mono">source=fallback</code>.
        </div>`}
      </div>
    `);
  }

  const t = data.totals || {};
  const n = (data.calls || []).length;

  return card(html`
    <div>
      ${fallbackBanner(data)}
      <div class="stats" style="margin-bottom:8px">
        ${stat('calls', t.n_live_calls)}
        ${stat('cached', t.n_cached_calls)}
        ${stat('latency', t.mean_latency_ms == null ? null : `${Math.round(t.mean_latency_ms)}ms`)}
      </div>
      <button class="btn" style="width:100%" onClick=${onOpen}>
        ${n > 0 ? `Read all ${n} exchanges` : 'Open LLM log'}
      </button>
    </div>
  `);
}

/* --- the drawer ------------------------------------------------------------ */

export function LlmLogDrawer({ runId, nEvents, onClose }) {
  // Prompts load here and only here. Opening this is the act of asking for them, so there is
  // no checkbox to find first — the summary card never needs them.
  const { data, err } = useLlmLog(runId, nEvents, true);
  const [sel, setSel] = useState(null);
  const listRef = useRef(null);

  const calls = data?.calls || [];
  // Default to the newest exchange: during a live run that is the one you came to read.
  const current = calls.find((c) => c.step === sel) || calls[calls.length - 1] || null;
  const steps = calls.map((c) => c.step);

  // The newest exchange is selected on open, so the list has to be looking at it. Without
  // this the detail pane shows step 10 while the list sits at step 1, which reads as a bug.
  useEffect(() => {
    const el = listRef.current?.querySelector('.lcall.sel');
    if (el) el.scrollIntoView({ block: 'nearest' });
  }, [current?.step, calls.length]);

  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') {
        onClose();
        return;
      }
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
      if (!steps.length) return;
      // The step is computed from the *current* selection inside the updater rather than
      // from one captured when this listener was registered. Holding the key down fires
      // faster than the re-render that would re-register it, and a stale closure recomputes
      // the same destination -- which presents as the arrow keys intermittently sticking.
      setSel((prev) => {
        const i = steps.indexOf(prev ?? steps[steps.length - 1]);
        const next =
          e.key === 'ArrowUp' ? Math.max(0, i - 1) : Math.min(steps.length - 1, i + 1);
        return steps[next];
      });
      e.preventDefault();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [steps.join(','), onClose]);

  return html`
    <div class="scrim" onClick=${onClose}></div>
    <aside class="drawer" role="dialog" aria-label="LLM log">
      <header>
        <b>LLM log</b>
        <span class="small muted">Prompts, decisions, and model usage</span>
        <span class="spacer"></span>
        <span class="small faint nowrap">
          ${calls.length > 1 ? '↑ ↓ to step · Esc to close' : 'Esc to close'}
        </span>
        <button class="btn ghost" onClick=${onClose} aria-label="Close">✕</button>
      </header>

      ${err && html`<div class="body"><div class="small" style="color:var(--err)">${err}</div></div>`}
      ${!data && !err && html`<div class="body"><div class="empty">loading…</div></div>`}

      ${data && data.uses_llm &&
      html`<div class="drawer-top">
        ${fallbackBanner(data)}
        <div class="stats" style="margin-bottom:0">
          ${stat('live calls', data.totals?.n_live_calls)}
          ${stat('from cache', data.totals?.n_cached_calls)}
          ${stat('input tok', data.totals?.input_tokens)}
          ${stat('output tok', data.totals?.output_tokens)}
          ${stat('cache reads', data.totals?.cache_read_input_tokens)}
          ${stat(
            'mean latency',
            data.totals?.mean_latency_ms == null
              ? null
              : `${Math.round(data.totals.mean_latency_ms)} ms`,
          )}
        </div>
        ${data.totals?.n_live_calls > 0 && !data.totals?.cache_read_input_tokens &&
        html`<div class="small" style="color:var(--warn);margin-top:8px">
          No cache reads across ${data.totals.n_live_calls} live calls. The stable prompt
          prefix is being invalidated, so every decision pays full input price.
        </div>`}
      </div>`}

      ${data && calls.length === 0 &&
      html`<div class="body"><div class="empty">
        ${data.uses_llm
          ? 'No LLM calls yet.'
          : `This run used the ${data.policy || 'rule-based'} policy, which makes no API calls.`}
      </div></div>`}

      ${calls.length > 0 &&
      html`<div class="drawer-body">
        <div class="drawer-list" ref=${listRef}>
          ${calls.map(
            (c) => html`
              <button
                type="button"
                key=${'l' + c.step}
                class=${'lcall' + (c.step === current?.step ? ' sel' : '')}
                onClick=${() => setSel(c.step)}
              >
                <span class="mono tiny muted" style="min-width:34px">#${c.step}</span>
                <span class="lcall-main">
                  <b class=${'sub-fg-' + c.subsystem}>${c.subsystem || c.action}</b>
                  <span class="lcall-sub small muted">${c.rationale || '—'}</span>
                </span>
                ${c.cached
                  ? html`<span class="pill">cached</span>`
                  : c.latency_ms
                    ? html`<span class="mono tiny faint nowrap">${Math.round(c.latency_ms)}ms</span>`
                    : null}
              </button>
            `,
          )}
        </div>

        <div class="drawer-detail">
          ${current && Exchange({ c: current })}
        </div>
      </div>`}
    </aside>
  `;
}

function Exchange({ c }) {
  return html`
    <div>
      <div class="row wrap" style="gap:8px;margin-bottom:10px">
        <span class="mono small muted">step ${c.step}</span>
        <b class=${'sub-fg-' + c.subsystem} style="font-size:15px">${c.subsystem || c.action}</b>
        ${c.cached && html`<span class="pill">cached</span>`}
      </div>

      ${c.rationale && html`<p class="read-lead">${c.rationale}</p>`}
      ${c.expected_gain &&
      html`<div class="small muted" style="margin-bottom:10px">
        Expected gain: ${c.expected_gain}
      </div>`}

      ${c.considered?.length > 0 &&
      html`<div style="margin-bottom:14px">
        <div class="lbl small muted" style="margin-bottom:5px">Alternatives it weighed</div>
        ${c.considered.map(
          (a, i) => html`<div class="small" key=${'x' + i} style="margin-bottom:2px">
            <span class="mono">${a.score?.toFixed ? a.score.toFixed(2) : a.score}</span>
            <b style="margin:0 6px" class=${'sub-fg-' + a.subsystem}>${a.subsystem}</b>
            <span class="muted">${a.reason}</span>
          </div>`,
        )}
      </div>`}

      <dl class="kv" style="margin-bottom:14px">
        <dt>model</dt><dd class="mono">${c.model || '—'}</dd>
        <dt>effort</dt><dd class="mono">${c.effort || '—'}</dd>
        <dt>stop reason</dt><dd class="mono">${c.stop_reason || '—'}</dd>
        <dt>tokens</dt>
        <dd class="mono">
          ${c.input_tokens ? `${c.input_tokens} in → ${c.output_tokens} out` : '—'}
        </dd>
        <dt>cache key</dt><dd class="mono tiny">${c.key || '—'}</dd>
      </dl>

      ${c.prompt
        ? html`<details class="exch" open>
            <summary>Prompt sent</summary>
            <pre class="read">${c.prompt}</pre>
          </details>`
        : html`<div class="small faint">No prompt recorded for this exchange.</div>`}
      ${c.response &&
      html`<details class="exch" open>
        <summary>Raw response</summary>
        <pre class="read">${JSON.stringify(c.response, null, 2)}</pre>
      </details>`}
    </div>
  `;
}

/* --- shared ---------------------------------------------------------------- */

function useLlmLog(runId, nEvents, full) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    let alive = true;
    getLlmLog(runId, full)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setErr(e.message));
    return () => {
      alive = false;
    };
    // nEvents re-fetches as the run progresses, so the log fills in live.
  }, [runId, full, nEvents]);
  return { data, err };
}

export function fallbackCount(data) {
  return data?.n_fallback_decisions || 0;
}

function fallbackBanner(data) {
  if (data.n_fallback_decisions > 0) {
    return html`<div class="banner warn">
      <b>${data.n_fallback_decisions} of ${data.n_decisions} decisions fell back to the
      scripted policy.</b> Those steps were not made by the LLM, so this run is not a clean
      measurement of the LLM arm.
      ${data.n_fallback_decisions === data.n_decisions &&
      html`<div style="margin-top:5px">
        Every decision fell back and no exchange was recorded, so the model was never reached.
        Check the API key and the server log before drawing conclusions from this run.
      </div>`}
    </div>`;
  }
  if ((data.calls || []).length > 0) {
    return html`<div class="banner ok">
      No fallbacks. Every decision in this run was made by the model.
    </div>`;
  }
  return null;
}

function stat(label, value) {
  return html`<div class="stat">
    <div class="v mono">${value == null ? '—' : value}</div>
    <div class="k small muted">${label}</div>
  </div>`;
}

function card(children) {
  return html`<div class="card">
    <h3>LLM log</h3>
    <div class="body">${children}</div>
  </div>`;
}
