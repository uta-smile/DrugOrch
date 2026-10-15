/* The policy card: what an arm does, what it reads, what it emits, and a worked example.
 *
 * This exists because choosing the routing policy is the one decision the system hands back
 * to the human, and a choice you cannot inspect is not really a choice. The card is rendered
 * from `/api/policies`, where the rule-based examples are produced by *running* each policy
 * on one shared synthetic state — so the example on screen cannot drift from the code, and
 * two arms can be compared by reading their outputs on the identical situation.
 *
 * The input/output pair is the part that earns its space. Prose about "adaptive routing"
 * tells a scientist nothing they can check; `step 7, 40 unscored, stalled 2 rounds → dock 8`
 * tells them exactly what they are about to run.
 */

import { useState } from 'preact/hooks';
import { html } from '@/h.js';

export function PolicyCard({ card, open, onToggle }) {
  if (!card) return null;
  return html`
    <div class=${'pcard' + (open ? ' open' : '')}>
      <button class="pcard-head" type="button" onClick=${onToggle}
              aria-expanded=${open ? 'true' : 'false'}>
        <i class=${'dot pol-' + card.name}></i>
        <b>${card.title}</b>
        <span class="small muted">${card.one_line}</span>
        <span class="spacer"></span>
        ${card.calls_api && html`<span class="pill warn-pill">calls the API</span>`}
        <span class="chev">${open ? '▾' : '▸'}</span>
      </button>

      ${open &&
      html`<div class="pcard-body">
        <div class="role">${card.role}</div>

        ${Field({ label: 'How it decides', children: card.rule })}

        <div class="io">
          <div>
            <div class="lbl small muted">Reads</div>
            <ul class="taglist">
              ${card.reads.map((r) => html`<li key=${r} class="mono">${r}</li>`)}
            </ul>
          </div>
          <div>
            <div class="lbl small muted">Emits</div>
            <ul class="taglist">
              ${card.emits.map((r) => html`<li key=${r} class="mono">${r}</li>`)}
            </ul>
          </div>
        </div>

        ${Example({ example: card.example })}

        <dl class="kv" style="margin-top:12px">
          <dt>stands for</dt><dd>${card.pattern}</dd>
          <dt>in the literature</dt><dd>${card.reference}</dd>
          <dt>reproducible</dt><dd>${card.determinism}</dd>
          <dt>use it</dt><dd>${card.use_when}</dd>
        </dl>

        <div class="caveat"><b>Watch out:</b> ${card.caveat}</div>
      </div>`}
    </div>
  `;
}

function Field({ label, children }) {
  return html`<div style="margin-bottom:11px">
    <div class="lbl small muted">${label}</div>
    <div class="small">${children}</div>
  </div>`;
}

function Example({ example }) {
  if (!example || example.error) {
    return html`<div class="small muted" style="margin-top:10px">
      No worked example available${example?.error ? ` (${example.error})` : ''}.
    </div>`;
  }
  return html`
    <div class="example">
      <div class="lbl small muted">
        Worked example
        <span class=${'pill ' + (example.computed ? 'done' : '')} style="margin-left:6px">
          ${example.computed ? 'computed by running this policy' : 'recorded illustration'}
        </span>
      </div>
      <div class="small muted" style="margin:5px 0 9px">${example.situation}</div>

      <div class="io">
        <div>
          <div class="lbl small muted">Input: what it sees</div>
          <pre class="mono tiny">${JSON.stringify(example.input, null, 1)}</pre>
        </div>
        <div>
          <div class="lbl small muted">Output: what it decides</div>
          ${DecisionBlock({ d: example.output })}
        </div>
      </div>
      ${example.note && html`<div class="small faint" style="margin-top:7px">${example.note}</div>`}
    </div>
  `;
}

function DecisionBlock({ d }) {
  if (!d || !d.action) return html`<div class="small muted">—</div>`;
  return html`
    <div class="dblock">
      <div class="row" style="gap:6px;margin-bottom:5px">
        <span class="pill">${d.action}</span>
        ${d.subsystem && html`<b class=${'sub-name sub-fg-' + d.subsystem}>${d.subsystem}</b>`}
      </div>
      ${d.params && Object.keys(d.params).length > 0 &&
      html`<div class="mono tiny muted" style="margin-bottom:5px">
        ${Object.entries(d.params).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join('  ')}
      </div>`}
      ${d.rationale && html`<div class="small">${d.rationale}</div>`}
      ${d.considered?.length > 0 &&
      html`<div style="margin-top:7px">
        <div class="lbl small muted">Rejected</div>
        ${d.considered.map(
          (a, i) => html`<div class="small" key=${'a' + i} style="margin-top:3px">
            <span class="mono">${a.score?.toFixed ? a.score.toFixed(2) : a.score}</span>
            <b style="margin:0 5px">${a.subsystem}</b>
            <span class="muted">${a.reason}</span>
          </div>`,
        )}
      </div>`}
    </div>
  `;
}

/** The full set, with one card expanded at a time. */
export function PolicyCatalog({ cards, initial = null }) {
  const [open, setOpen] = useState(initial);
  if (!cards?.length) return html`<div class="empty">loading policy descriptions…</div>`;
  return html`
    <div class="pcards">
      ${cards.map(
        (c) => html`<${PolicyCard} key=${c.name} card=${c} open=${open === c.name}
                                   onToggle=${() => setOpen(open === c.name ? null : c.name)} />`,
      )}
    </div>
  `;
}
