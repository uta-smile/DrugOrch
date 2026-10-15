/* Which subsystems this run is given.
 *
 * Unticking a tool removes it from the registry the run is built with, which means it is
 * absent from the capability manifest and from the JSON schema the model answers under — so
 * it cannot be chosen rather than being chosen and then refused. That distinction is why
 * this is a run-creation control and not a filter on the trace.
 *
 * The parameters shown under each tool are read from the capability manifest, the same
 * declaration that drives the model's schema and the clamping. They are shown here because
 * "the policy chose n_actives=100" is only meaningful next to the range it was choosing
 * from; the values are the subsystem's defaults and the policy may move them within those
 * bounds at any step.
 */

import { useState } from 'preact/hooks';
import { html } from '@/h.js';

export function ToolPicker({ subsystems, enabled, onChange }) {
  const [open, setOpen] = useState(null);
  const on = new Set(enabled);

  function toggle(key) {
    const next = new Set(on);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    // Keep manifest order rather than click order: this list is what the run record shows.
    onChange(subsystems.map((s) => s.key).filter((k) => next.has(k)));
  }

  return html`
    <div class="tools">
      ${subsystems.map((s) => {
        const sel = on.has(s.key);
        const isOpen = open === s.key;
        return html`
          <div class=${'tool' + (sel ? ' on' : '')} key=${s.key}>
            <label class="tool-head">
              <input type="checkbox" checked=${sel} onChange=${() => toggle(s.key)} />
              <i class=${'dot sub-' + s.key}></i>
              <span class="n">${s.key}</span>
              <span class="small muted nowrap">
                ${s.unit_cost}u / ${s.per} · ~${Math.round(s.typical_wall_s)}s
              </span>
              <span class="spacer"></span>
              <button type="button" class="q" title="Tool description and parameters"
                      onClick=${(e) => { e.preventDefault(); setOpen(isOpen ? null : s.key); }}>
                ${isOpen ? '−' : '?'}
              </button>
            </label>
            ${isOpen &&
            html`<div class="tool-body">
              <p class="small">${s.summary}</p>
              ${s.requires?.length > 0 &&
              html`<p class="small muted">Needs first: ${s.requires.join(', ')}</p>`}
              ${s.produces?.length > 0 &&
              html`<p class="small muted">Produces: ${s.produces.join(', ')}</p>`}
              ${s.params?.length > 0 &&
              html`<table class="cd-table">
                <thead><tr><th>parameter</th><th>range</th><th>default</th><th></th></tr></thead>
                <tbody>
                  ${s.params.map(
                    (p) => html`<tr key=${p.name}>
                      <td class="mono">${p.name}</td>
                      <td class="mono small">${range(p)}</td>
                      <td class="mono small">${fmtDefault(p.default)}</td>
                      <td class="small muted">${p.doc}</td>
                    </tr>`,
                  )}
                </tbody>
              </table>`}
              <p class="small faint">
                The routing policy chooses these within the ranges shown, every step. What it
                actually chose appears on each block in the trace.
              </p>
            </div>`}
          </div>
        `;
      })}
      ${on.size === 0 &&
      html`<p class="note err">A run needs at least one tool.</p>`}
    </div>
  `;
}

function range(p) {
  if (p.choices?.length) return p.choices.join(' | ');
  if (p.lo != null || p.hi != null) return `${p.lo ?? '−∞'} … ${p.hi ?? '∞'}`;
  return p.type;
}

function fmtDefault(v) {
  if (Array.isArray(v)) return v.length ? `[${v.length}]` : '[]';
  return String(v);
}
