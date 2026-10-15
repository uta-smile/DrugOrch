/* What each side of the public benchmark is allowed to do.
 *
 * A success rate is not interpretable on its own. MolDesignBench gives the models it
 * evaluates seventeen tools; DrugOrch answers the same instances with four subsystems. Those
 * are different experiments, and a leaderboard that reports only the scores invites the
 * reader to conclude something about the models that the numbers do not support.
 *
 * So the two rosters are shown side by side, every tool carries what it actually does, and
 * the capabilities only one side has are named rather than left to be inferred from a count.
 * The benchmark's descriptions are the strings its own agents are shown, read from its
 * sources rather than paraphrased here: a paraphrase of a tool description is a claim about
 * someone else's system.
 *
 * The page is deliberately dense. It is reference material, read once by someone deciding
 * whether the comparison is fair, not a dashboard.
 */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { getBenchTools } from '@/api.js';

const ORDER = ['measurement', 'retrieval', 'generation', 'representation', 'general'];

export function ToolsView() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [open, setOpen] = useState(null);

  useEffect(() => {
    getBenchTools().then(setData).catch((e) => setErr(e.message));
  }, []);

  if (err) return html`<div class="card"><div class="body"><div class="note err">${err}</div></div></div>`;
  if (!data) return html`<div class="empty">loading…</div>`;

  const byGroup = (side) => {
    const out = {};
    for (const t of data[side] || []) (out[t.group] ||= []).push(t);
    return out;
  };
  const theirs = byGroup('benchmark');
  const ours = byGroup('drugorch');
  const groups = ORDER.filter((g) => theirs[g]?.length || ours[g]?.length);

  return html`
    <div class="tools-page">
      <div class="bench-head">
        <div>
          <h2>Toolsets</h2>
          <p class="muted small">
            MolDesignBench provides ${data.n_benchmark} tools; DrugOrch uses
            ${' '}${data.n_drugorch} subsystems on the same instances. Expand an item for its
            description and parameters.
          </p>
        </div>
        <span class="spacer"></span>
        <div class="bench-winner">
          <div class="k">tools available</div>
          <div class="v">${data.n_drugorch} <span class="small muted" style="font-weight:400">
            against ${data.n_benchmark}</span></div>
        </div>
      </div>

      <div class="note" style="margin-bottom:16px">
        MolDesignBench includes name converters and a Python shell. Capabilities without a
        DrugOrch counterpart are listed below.
      </div>

      ${groups.map(
        (g) => html`<section class="tool-group" key=${g}>
          <h3>${data.groups?.[g] || g}</h3>
          <div class="tool-cols">
            <div>
              <div class="side-head mine">DrugOrch</div>
              ${(ours[g] || []).map((t) => html`<${Tool} key=${t.name} t=${t}
                                                        open=${open} setOpen=${setOpen} />`)}
              ${!(ours[g] || []).length &&
              html`<div class="tool-none">No tool in this category.</div>`}
            </div>
            <div>
              <div class="side-head">MolDesignBench</div>
              ${(theirs[g] || []).map((t) => html`<${Tool} key=${t.name} t=${t}
                                                           open=${open} setOpen=${setOpen} />`)}
              ${!(theirs[g] || []).length &&
              html`<div class="tool-none">No tool in this category.</div>`}
            </div>
          </div>
        </section>`,
      )}

      <section class="tool-group">
        <h3>Benchmark capabilities without a DrugOrch counterpart</h3>
        <p class="muted small" style="margin:0 0 10px">
          ${data.uncovered?.length} of ${data.n_benchmark} benchmark tools have no counterpart
          ${' '}in DrugOrch. These tools affect what each model can do on a task.
        </p>
        <ul class="gap-list">
          ${(data.uncovered || []).map(
            (u) => html`<li key=${u.name}>
              <span class="mono">${u.name}</span>
              <span class="tag">${data.groups?.[u.group] || u.group}</span>
              <div class="small muted">${u.description.slice(0, 220)}${u.description.length > 220 ? '…' : ''}</div>
            </li>`,
          )}
        </ul>
      </section>

      <p class="small muted tool-foot">${data.note}</p>
    </div>
  `;
}

/* One tool. Collapsed to its name and first sentence, because seventeen expanded
 * descriptions is a wall; expanded on click, because the detail is the point of the page. */
function Tool({ t, open, setOpen }) {
  const id = t.side + ':' + t.name;
  const isOpen = open === id;
  const first = t.description.split('. ')[0];

  return html`<div class=${'tool-card' + (t.side === 'drugorch' ? ' mine' : '') + (isOpen ? ' open' : '')}>
    <button class="tool-head" onClick=${() => setOpen(isOpen ? null : id)}
            aria-expanded=${isOpen}>
      <span class="mono name">${t.name}</span>
      <span class="chev">${isOpen ? '−' : '+'}</span>
    </button>
    ${!isOpen && html`<p class="tool-sum">${first}${first.endsWith('.') ? '' : '.'}</p>`}
    ${isOpen &&
    html`<div class="tool-body">
      <p>${t.description}</p>
      ${t.detail && html`<p class="detail"><b>How it works here.</b> ${t.detail}</p>`}
      ${Object.keys(t.parameters || {}).length > 0 &&
      html`<dl class="params">
        ${Object.entries(t.parameters).map(
          ([k, v]) => html`<div key=${k}>
            <dt class="mono">${k}</dt>
            <dd>${v || 'no description given'}</dd>
          </div>`,
        )}
      </dl>`}
      <p class="small muted">
        ${t.counterpart && html`<span>Stands in for: ${t.counterpart}. </span>`}
        <span class="mono tiny">${t.source}</span>
      </p>
    </div>`}
  </div>`;
}
