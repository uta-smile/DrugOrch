/* The selected decision, in full: why it was taken, what it beat, and what it knew.
 *
 * This is where the trace stops being a picture of activity and becomes an argument. The
 * rejected-alternatives table is the point: a reader can see that the policy weighed docking
 * at 0.75 and generation at 0.10 and say whether they agree -- which is not a judgement any
 * aggregate metric lets them make.
 *
 * "What the policy saw" closes the loop. A rationale is the policy's account of itself and
 * can be wrong or self-serving; `state_seen` is the actual input, fetched on demand because
 * it is a few KB per step. With both, disagreeing with a decision is a checkable claim --
 * the scientist can see the policy had 40 unscored molecules in front of it and still chose
 * to generate more, and go change the rule that did that.
 */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { getDecisions } from '@/api.js';
import { CallDetail } from '@/components/CallDetail.js';

export function Inspector({ tick, block, capabilities, runId }) {
  if (!tick) {
    return html`<div class="empty">
      Select a decision in the trace above to see why it was made.
    </div>`;
  }

  const chosen = tick.action === 'stop' ? null : tick.subsystem;
  const rows = [...(tick.considered || [])]
    .filter((a) => a.subsystem && a.subsystem !== chosen)
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

  const cap = chosen ? capabilities?.[chosen] : null;

  return html`
    <div class="inspector">
      <div class="hdr">
        <span class="step">step ${tick.step}</span>
        ${chosen
          ? html`<span class="row">
              <i class=${'dot sub-' + chosen}></i>
              <b>${chosen}</b>
            </span>`
          : html`<b>stop</b>`}
        <span class="pill">${tick.policy || 'policy'}</span>
        ${tick.source === 'llm' && html`<span class="pill running">llm</span>`}
        ${tick.source === 'fallback' &&
        html`<span class="pill failed" title="the policy under test failed; a fallback decided">
          fallback
        </span>`}
        ${tick.blocked &&
        html`<span class="pill failed" title=${tick.blockedReason}>budget blocked</span>`}
        <span class="spacer" style="flex:1"></span>
        ${tick.latencyMs > 0 &&
        html`<span class="small faint mono">${Math.round(tick.latencyMs)} ms to decide</span>`}
      </div>

      <p class="why">${tick.rationale || html`<i class="muted">no rationale recorded</i>`}</p>
      ${tick.expectedGain &&
      html`<div class="gain"><b>Expected gain:</b> ${tick.expectedGain}</div>`}

      ${rows.length > 0 &&
      html`
        <table class="alts">
          <thead>
            <tr>
              <th class="mark"></th>
              <th>option</th>
              <th class="score">score</th>
              <th>reasoning</th>
            </tr>
          </thead>
          <tbody>
            ${chosen &&
            html`<tr class="chosen">
              <td class="mark" aria-label="chosen">✓</td>
              <td class="sname"><i class=${'dot sub-' + chosen}></i> ${chosen}</td>
              <td class="score">—</td>
              <td>chosen</td>
            </tr>`}
            ${rows.map(
              (a, i) => html`
                <tr key=${'a' + i}>
                  <td class="mark faint" aria-label="rejected">✕</td>
                  <td class="sname faint">
                    <i class=${'dot sub-' + a.subsystem} style="opacity:.45"></i> ${a.subsystem}
                  </td>
                  <td class="score faint">${(a.score ?? 0).toFixed(2)}</td>
                  <td class="muted">${a.reason || '—'}</td>
                </tr>
              `,
            )}
          </tbody>
        </table>
      `}

      ${block &&
      html`
        <div class="io">
          <div>
            <span class="k">in</span>
            <span class="mono">${formatParams(block.params, cap)}</span>
          </div>
          <div style="margin-top:4px">
            <span class="k">out</span>
            ${block.running
              ? html`<span class="muted">running… ${block.message || ''}</span>`
              : html`<span class=${block.ok ? '' : 'mono'} style=${block.ok ? '' : 'color:var(--err)'}>
                  ${block.ok ? block.note : block.error}
                </span>`}
          </div>
          ${!block.running &&
          html`<div class="small faint" style="margin-top:4px">
            ${(block.cost ?? block.w).toFixed(2)} units · ${block.wallS.toFixed(1)} s
            ${block.cacheHits > 0 ? ` · ${block.cacheHits} from cache` : ''}
          </div>`}
          ${block.top?.length > 0 &&
          html`<div class="small" style="margin-top:6px">
            <span class="k">best</span>
            ${block.top.slice(0, 3).map(
              (t, i) => html`<span key=${'t' + i} class="mono" style="margin-right:10px">
                ${fmtScore(t)}
              </span>`,
            )}
          </div>`}
        </div>
      `}

      ${block?.explain &&
      html`<${CallDetail} explain=${block.explain} subsystem=${block.subsystem} />`}

      ${runId && html`<${StateSeen} runId=${runId} step=${tick.step} />`}
    </div>
  `;
}

function StateSeen({ runId, step }) {
  const [state, setState] = useState(null);
  const [err, setErr] = useState(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open || !runId || !step) return undefined;
    let alive = true;
    setState(null);
    setErr(null);
    getDecisions(runId, step)
      .then((d) => alive && setState(d.decisions?.[0]?.state_seen ?? {}))
      .catch((e) => alive && setErr(e.message));
    return () => {
      alive = false;
    };
  }, [runId, step, open]);

  return html`
    <div class="seen">
      <button class="btn ghost small" type="button" onClick=${() => setOpen(!open)}>
        ${open ? 'hide' : 'show'} what the policy saw
      </button>
      ${open && err && html`<div class="small" style="color:var(--err)">${err}</div>`}
      ${open && !state && !err && html`<div class="small muted">loading…</div>`}
      ${open && state &&
      html`<div>
        <div class="small muted" style="margin:7px 0 5px">
          The exact <code class="mono">state.summary()</code> passed to the policy at this
          step. Every arm is offered this identical view.
        </div>
        <pre class="mono tiny scroller">${JSON.stringify(state, null, 1)}</pre>
      </div>`}
    </div>
  `;
}

/* Parameters that are never noise, even at their default value: each names *what was asked
 * for* rather than tuning how hard to ask. `fetch=all` and `query_from=generated` are the
 * whole content of a retrieval or similarity call — hiding them because they happened to be
 * the default left the trace saying "retrieval ran, defaults", which is the opacity these
 * panels exist to remove. */
const ALWAYS_SHOW = new Set([
  'backend',
  'acquisition',
  'selector',
  'fetch',
  'query_from',
  'source',
  'safety',
]);

function formatParams(params, cap) {
  if (!params || !Object.keys(params).length) return 'defaults';
  const defaults = Object.fromEntries((cap?.params || []).map((p) => [p.name, p.default]));
  const parts = [];
  for (const [k, v] of Object.entries(params)) {
    if (Array.isArray(v)) {
      if (v.length) parts.push(`${k}=[${v.length} molecules]`);
      continue;
    }
    // Only show what the policy actually chose to change -- except the backend and the
    // acquisition function, which are never noise: they say which algorithm ran.
    if (!ALWAYS_SHOW.has(k) && JSON.stringify(defaults[k]) === JSON.stringify(v)) continue;
    parts.push(`${k}=${v}`);
  }
  return parts.length ? parts.join('  ') : 'defaults';
}

function fmtScore(t) {
  const key = Object.keys(t).find((k) => k !== 'smiles');
  const v = t[key];
  return `${typeof v === 'number' ? v.toFixed(2) : v}`;
}
