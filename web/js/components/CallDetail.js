/* What a subsystem actually did, rendered per subsystem.
 *
 * A block in the trace used to say how much it cost and one sentence of prose. That answers
 * "did it run" but not "is it working" — a generation round that bred on nothing and one
 * that bred on real docking scores looked identical, and a GA saturating against its own
 * parent pool looked like a GA doing fine.
 *
 * Each renderer reads the `explain` blob its subsystem builds, in that subsystem's own
 * vocabulary. An unknown key falls through to a plain table, so a subsystem added later is
 * explainable here without this file changing.
 */

import { useState } from 'preact/hooks';
import { html } from '@/h.js';
import { depictUrl } from '@/api.js';

export function CallDetail({ explain, subsystem }) {
  if (!explain || !Object.keys(explain).length) return null;
  const Body = RENDERERS[subsystem] || Generic;
  return html`
    <div class="calldetail">
      ${explain.rationale && html`<p class="cd-why">${explain.rationale}</p>`}
      <${Body} e=${explain} />
    </div>
  `;
}

/* --- generation ----------------------------------------------------------- */
function Generation({ e }) {
  const rejected = Object.entries(e.rejected || {}).filter(([, n]) => n > 0);
  const attempts = e.attempts || 0;
  const accepted = (e.accepted || []).length;
  // The yield is the GA's own health reading: a round that tried 240 times for 20 children
  // is nearly exhausted, whatever its note says.
  const yieldPct = attempts ? Math.round((accepted / attempts) * 100) : null;

  return html`
    <div class="cd">
      <div class="cd-row">
        ${stat('generation', e.generation)}
        ${stat('backend', e.backend)}
        ${stat('bred on', e.parent_basis, e.parent_basis === 'insertion_order'
          ? 'No fitness was available. These parents were taken in order, not selected.'
          : `Parents ranked by ${e.parent_basis}.`)}
        ${stat('yield', yieldPct == null ? '—' : `${accepted}/${attempts} (${yieldPct}%)`)}
      </div>

      ${e.parent_basis === 'insertion_order' &&
      html`<p class="cd-warn">
        No scores existed, so parents were taken in insertion order. This round bred on
        nothing. Retrieve actives or dock a compound first.
      </p>`}

      ${e.parents?.length > 0 &&
      html`<${Section} title=${`Parents (${e.parents.length})`}>
        <table class="cd-table">
          <thead><tr><th>#</th><th>fitness</th><th>molecule</th></tr></thead>
          <tbody>
            ${e.parents.map(
              (p) => html`<tr key=${p.smiles}>
                <td class="num">${p.rank}</td>
                <td class="num mono">${p.fitness == null ? '—' : p.fitness.toFixed(2)}</td>
                <td class="smi mono">${p.smiles}</td>
              </tr>`,
            )}
          </tbody>
        </table>
      <//>`}

      ${(Object.keys(e.operators || {}).length > 0 || rejected.length > 0) &&
      html`<${Section} title="Operators and rejections">
        <div class="cd-chips">
          ${Object.entries(e.operators || {}).map(
            ([k, n]) => n > 0 && html`<span class="chip ok" key=${k}>${n} ${k}</span>`,
          )}
          ${rejected.map(
            ([k, n]) => html`<span class="chip" key=${k} title=${REJECT_WHY[k] || ''}>
              ${n} ${k}
            </span>`,
          )}
        </div>
        ${Object.keys(e.mutations_used || {}).length > 0 &&
        html`<div class="cd-smarts">
          ${Object.entries(e.mutations_used).map(
            ([smarts, n]) => html`<div key=${smarts}>
              <span class="mono">${smarts}</span> <span class="muted">x${n}</span>
            </div>`,
          )}
        </div>`}
      <//>`}

      ${e.accepted?.length > 0 &&
      html`<${Section} title=${`Offspring (${e.accepted.length})`}>
        <div class="cd-mols">
          ${e.accepted.slice(0, 8).map(
            (c) => html`<div class="cd-mol" key=${c.smiles}>
              <img src=${depictUrl(c.smiles, 150, 110)} alt="" loading="lazy"
                   width="100" height="74" />
              <div class="small muted">${c.operator}</div>
            </div>`,
          )}
        </div>
        ${e.accepted_truncated > 0 &&
        html`<div class="small faint">… ${e.accepted_truncated} more</div>`}
      <//>`}
    </div>
  `;
}

const REJECT_WHY = {
  invalid: 'The operator produced nothing that survived sanitization or the size filters.',
  duplicate: 'Already in the run. Mostly duplicates means the parent pool is exhausted.',
  scaffold: 'Rejected by the novel-scaffolds-only filter.',
  same_parent: 'The two parents drawn were the same molecule.',
};

/* --- retrieval ------------------------------------------------------------ */
function Retrieval({ e }) {
  const s = e.structures;
  const a = e.actives;
  return html`
    <div class="cd">
      <div class="cd-row">
        ${stat('ran', (e.ran || []).join(' · ') || '—')}
        ${stat('requests', e.banks?.requests)}
        ${stat('from cache', e.banks?.from_cache)}
      </div>

      ${e.identity &&
      html`<${Section} title="Target">
        <dl class="kv">
          <dt>resolved</dt><dd class="mono">${e.identity.resolved_to} · ${e.identity.gene}</dd>
          <dt>organism</dt><dd>${e.identity.organism}</dd>
          <dt>length</dt><dd>${e.identity.sequence_length} aa</dd>
          <dt>why</dt><dd>${e.identity.why}</dd>
        </dl>
      <//>`}

      ${s &&
      html`<${Section} title=${`Structures: ${s.examined} examined, ${s.dockable} dockable`}>
        ${s.selection && html`<p class="cd-why">${s.selection}</p>`}
        <table class="cd-table">
          <thead><tr><th></th><th>PDB</th><th>Å</th><th>ligand</th><th>score</th><th>why</th></tr></thead>
          <tbody>
            ${(s.ranking || []).map(
              (r) => html`<tr key=${r.pdb_id} class=${r.pdb_id === s.chosen ? 'cur' : ''}>
                <td>${r.pdb_id === s.chosen ? '✓' : ''}</td>
                <td class="mono">${r.pdb_id}</td>
                <td class="num mono">${r.resolution == null ? '—' : r.resolution.toFixed(2)}</td>
                <td class="mono">${r.ligand || '—'}</td>
                <td class="num mono">${r.score}</td>
                <td class="small muted">${r.reason}</td>
              </tr>`,
            )}
          </tbody>
        </table>
      <//>`}

      ${a &&
      html`<${Section} title=${`Actives: ${a.returned} of ${a.requested} requested`}>
        ${a.selection && html`<p class="cd-why">${a.selection}</p>`}
        ${a.withheld_by_holdout > 0 &&
        html`<p class="cd-warn">
          ${a.withheld_by_holdout} molecules were withheld by the benchmark holdout, so this
          run has not seen them. The rediscovery figure depends on that.
        </p>`}
        <div class="cd-chips">
          <span class="chip">${a.filter}</span>
          ${a.pchembl_range &&
          html`<span class="chip">pChEMBL ${a.pchembl_range[0]}–${a.pchembl_range[1]}</span>`}
          ${Object.entries(a.by_assay || {}).map(
            ([k, n]) => html`<span class="chip" key=${k} title="assay type">${n} ${k}</span>`,
          )}
        </div>
        <table class="cd-table" style="margin-top:7px">
          <thead><tr><th>pChEMBL</th><th>assay</th><th>ChEMBL</th><th>molecule</th></tr></thead>
          <tbody>
            ${(a.sample || []).map(
              (m) => html`<tr key=${m.chembl_id}>
                <td class="num mono">${m.pchembl}</td>
                <td class="mono">${m.assay}</td>
                <td class="mono small">${m.chembl_id}</td>
                <td class="smi mono">${m.smiles}</td>
              </tr>`,
            )}
          </tbody>
        </table>
        ${a.sample_truncated > 0 &&
        html`<div class="small faint">
          … ${a.sample_truncated} more of the ${a.returned} retrieved. The table is a sample;
          all of them entered the run.
        </div>`}
        ${(a.sample || []).length > 0 &&
        html`<div class="cd-mols" style="margin-top:7px">
          ${a.sample.slice(0, 8).map(
            (m) => html`<div class="cd-mol" key=${'d' + m.chembl_id}>
              <img src=${depictUrl(m.smiles, 150, 110)} alt="" loading="lazy"
                   width="100" height="74" />
              <div class="small muted">${m.pchembl}</div>
            </div>`,
          )}
        </div>`}
      <//>`}
    </div>
  `;
}

/* --- properties ------------------------------------------------------------
 *
 * "Profiled 20 molecules" is true and useless: it does not say which properties were looked
 * at, what the cutoffs were, or whether the answers were any good. So the panel itself is
 * the content here — every descriptor with its units and the threshold that turns it into a
 * flag — and the molecule table is secondary.
 */
function Properties({ e }) {
  const p = e.panel || {};
  const b = e.batch || {};
  const [open, setOpen] = useState(false);
  const mols = e.molecules || [];
  const shown = open ? mols : mols.slice(0, 6);

  return html`
    <div class="cd">
      <div class="cd-row">
        ${stat('profiled', b.profiled)}
        ${stat('computed', b.computed)}
        ${stat('from cache', b.from_cache)}
        ${stat('failed', b.failed || 0, b.failed ? b.failed_examples?.join('  ') : '')}
      </div>

      ${e.selection && html`<p class="cd-why">Looked at ${e.selection}.</p>`}

      <${Section} title=${`Panel: ${(p.descriptors || []).length} descriptors, ` +
                          `${(p.scores || []).length} scores, ${(p.flags || []).length} flags`}>
        <table class="cd-table">
          <thead><tr><th>property</th><th>min</th><th>median</th><th>max</th><th>rule</th></tr></thead>
          <tbody>
            ${(p.scores || []).map((d) => row(d, d.scale))}
            ${(p.descriptors || []).map((d) => row(d, d.threshold, d.unit))}
          </tbody>
        </table>
      <//>

      ${(p.flags || []).length > 0 &&
      html`<${Section} title="Developability flags">
        ${p.flags.map(
          (f) => html`<div class="cd-flag" key=${f.key} title=${f.rule}>
            <span class=${'chip' + (f.n_tripped ? '' : ' ok')}>
              ${f.n_tripped} / ${f.of}
            </span>
            <span><b>${f.title}</b> <span class="muted small">${f.rule}</span></span>
          </div>`,
        )}
      <//>`}

      ${e.pains?.length > 0 &&
      html`<${Section} title=${`PAINS alerts (${e.pains.length})`}>
        ${e.pains.map(
          (m) => html`<div class="small" key=${m.smiles}>
            <span class="mono">${m.alert}</span>
            <span class="smi mono muted" style="margin-left:6px">${m.smiles}</span>
          </div>`,
        )}
      <//>`}

      ${mols.length > 0 &&
      html`<${Section} title=${`Molecules, best QED first (${mols.length})`}>
        <table class="cd-table">
          <thead><tr>
            <th>QED</th><th>SA</th><th>MW</th><th>cLogP</th><th>TPSA</th>
            <th>Ro5</th><th>molecule</th>
          </tr></thead>
          <tbody>
            ${shown.map(
              (m) => html`<tr key=${m.smiles} class=${m.pains_alert ? 'warn' : ''}>
                <td class="num mono">${num(m.qed)}</td>
                <td class="num mono">${num(m.sa_score)}</td>
                <td class="num mono">${num(m.mw, 0)}</td>
                <td class="num mono">${num(m.logp)}</td>
                <td class="num mono">${num(m.tpsa, 0)}</td>
                <td class="num mono" title=${m.veber_pass === false ? 'fails Veber' : ''}>
                  ${m.lipinski_violations ?? '—'}${m.veber_pass === false ? '*' : ''}
                </td>
                <td class="smi mono">${m.smiles}</td>
              </tr>`,
            )}
          </tbody>
        </table>
        ${mols.length > 6 &&
        html`<button class="btn ghost small" onClick=${() => setOpen(!open)}>
          ${open ? 'show fewer' : `show all ${mols.length}`}
        </button>`}
        ${e.molecules_truncated > 0 &&
        html`<div class="small faint">… ${e.molecules_truncated} more not carried on the event</div>`}
      <//>`}
    </div>
  `;

  function row(d, rule, unit) {
    return html`<tr key=${d.key}>
      <td title=${d.key}>${d.title}${unit ? html` <span class="muted">(${unit})</span>` : ''}</td>
      <td class="num mono">${num(d.spread?.min)}</td>
      <td class="num mono">${num(d.spread?.median)}</td>
      <td class="num mono">${num(d.spread?.max)}</td>
      <td class="small muted">${rule || ''}</td>
    </tr>`;
  }
}

/* --- similarity ------------------------------------------------------------
 *
 * Two things a reader has to be able to check: which molecules went out to the bank and on
 * what basis, and what was turned away before anything entered the run. Both are stated
 * rather than left to be inferred from a count — that opacity is why this panel exists.
 */
function Similarity({ e }) {
  const q = e.queries || {};
  const s = e.search || {};
  const safety = e.safety || {};
  const m = e.metrics || {};
  // `duplicate` is counted apart from the rules: the run already had those molecules, which
  // is not the gate refusing anything. Folding it in would make the header disagree with
  // the table under it, where duplicate is labelled as not a safety rule.
  const refused = Object.entries(safety.rejected || {})
    .filter(([k, n]) => k !== 'duplicate' && n > 0)
    .reduce((a, [, n]) => a + n, 0);
  const kept = e.kept || [];

  return html`
    <div class="cd">
      <div class="cd-row">
        ${stat('kept', m.n_kept)}
        ${stat('raw hits', s.raw_hits)}
        ${stat('clinical', m.n_clinical, 'reached at least phase 1 in ChEMBL')}
        ${stat('approved', m.n_approved, 'max_phase 4')}
        ${stat('gate', safety.mode)}
      </div>

      <${Section} title=${`Searched with ${(q.used || []).length} molecules`}>
        <p class="cd-why">Chosen as: ${q.rule}.</p>
        <div class="cd-chips">
          ${Object.entries(q.available || {}).map(
            ([k, n]) => html`<span class="chip" key=${k} title="available to choose from">
              ${n} ${k}
            </span>`,
          )}
        </div>
        <table class="cd-table" style="margin-top:7px">
          <thead><tr><th>origin</th><th>hits</th><th>best</th><th>query molecule</th></tr></thead>
          <tbody>
            ${(q.used || []).map(
              (u) => html`<tr key=${u.smiles}>
                <td class="small"><i class=${'dot sub-' + u.origin}></i> ${u.origin}</td>
                <td class="num mono">${u.n_hits}</td>
                <td class="num mono">${num(u.best_tanimoto)}</td>
                <td class="smi mono">${u.smiles}</td>
              </tr>`,
            )}
          </tbody>
        </table>
      <//>

      <${Section} title="How it searched">
        <dl class="kv">
          <dt>banks</dt><dd class="mono">${(s.banks || []).join(', ')}</dd>
          <dt>metric</dt><dd>${s.metric}</dd>
          <dt>floor</dt><dd class="mono">Tanimoto ≥ ${s.min_similarity}</dd>
          ${s.dropped_over_cap > 0 &&
          html`<dt>over cap</dt><dd>${s.dropped_over_cap} dropped at max_results=${s.max_results}</dd>`}
        </dl>
      <//>

      <${Section} title=${`Safety gate: ${safety.mode}` +
                          (refused ? `, ${refused} turned away` : ', nothing turned away')}>
        <table class="cd-table">
          <thead><tr><th>rule</th><th>verdict</th><th>n</th><th>what it catches</th></tr></thead>
          <tbody>
            ${(safety.rules || []).map(
              (r) => html`<tr key=${r.key} class=${r.rejected ? 'warn' : ''}>
                <td>${r.title}</td>
                <td><span class=${'chip ' + (r.verdict === 'reject' ? '' : 'ok')}>${r.verdict}</span></td>
                <td class="num mono">${r.rejected || '—'}</td>
                <td class="small muted">${r.why}</td>
              </tr>`,
            )}
            ${safety.rejected?.duplicate > 0 &&
            html`<tr>
              <td>Already in the run</td>
              <td><span class="chip ok">skip</span></td>
              <td class="num mono">${safety.rejected.duplicate}</td>
              <td class="small muted">Not a safety rule. The run already held these.</td>
            </tr>`}
          </tbody>
        </table>
        ${safety.examples?.length > 0 &&
        html`<div class="cd-smarts" style="margin-top:6px">
          ${safety.examples.map(
            (x, i) => html`<div key=${'ex' + i}>
              <span class="chip">${x.rule}</span>
              <span class="muted small">${x.detail}</span>
              <span class="mono smi">${x.smiles}</span>
            </div>`,
          )}
        </div>`}
      <//>

      ${kept.length > 0 &&
      html`<${Section} title=${`Catalogued molecules admitted (${kept.length})`}>
        <table class="cd-table">
          <thead><tr>
            <th>T</th><th>phase</th><th>bank</th><th>QED</th><th>name / molecule</th>
          </tr></thead>
          <tbody>
            ${kept.map(
              (k) => html`<tr key=${k.smiles} class=${k.withdrawn || k.black_box ? 'warn' : ''}>
                <td class="num mono" title=${k.bank_similarity != null
                    ? `bank reported ${k.bank_similarity}`
                    : 'this bank reports no similarity of its own'}>${num(k.tanimoto)}</td>
                <td class="num mono" title=${k.first_approval ? `first approved ${k.first_approval}` : ''}>
                  ${k.max_phase == null ? '—' : k.max_phase}
                </td>
                <td class="mono small">${k.bank_id}</td>
                <td class="num mono">${num(k.qed)}</td>
                <td>
                  ${k.name && html`<b>${k.name}</b> `}
                  ${k.withdrawn && html`<span class="chip">withdrawn</span> `}
                  ${k.black_box && html`<span class="chip">black box</span> `}
                  ${(k.flags || []).map((f) => html`<span class="chip" key=${f}>${f}</span> `)}
                  <span class="smi mono muted">${k.smiles}</span>
                </td>
              </tr>`,
            )}
          </tbody>
        </table>
        ${e.kept_truncated > 0 &&
        html`<div class="small faint">… ${e.kept_truncated} more not carried on the event</div>`}
      <//>`}

      ${m.resemblance_to_actives != null &&
      html`<p class="cd-why" style="margin-top:8px">
        Mean similarity of these to the run's known actives is${' '}
        <b class="mono">${num(m.resemblance_to_actives)}</b>${'. '}
        ${m.resemblance_to_actives > 0.6
          ? 'the generator is circling chemistry the target already has.'
          : 'these sit away from the known actives, so this is new ground rather than rediscovery.'}
        ${m.diversity != null && html` Diversity among them is ${num(m.diversity)}.`}
      </p>`}
    </div>
  `;
}

function num(v, digits = 2) {
  return typeof v === 'number' ? v.toFixed(digits) : '—';
}

/* --- fallback: anything without a dedicated renderer ---------------------- */
function Generic({ e }) {
  const rows = Object.entries(e).filter(([k]) => k !== 'rationale');
  if (!rows.length) return null;
  return html`<div class="cd">
    <dl class="kv">
      ${rows.map(
        ([k, v]) => html`<dt key=${'k' + k}>${k}</dt><dd key=${'v' + k} class="mono small">
          ${typeof v === 'object' ? JSON.stringify(v) : String(v)}
        </dd>`,
      )}
    </dl>
  </div>`;
}

const RENDERERS = {
  generation: Generation,
  retrieval: Retrieval,
  properties: Properties,
  similarity: Similarity,
};

function Section({ title, children }) {
  return html`<details class="cd-sec" open>
    <summary>${title}</summary>
    <div class="cd-sec-body">${children}</div>
  </details>`;
}

function stat(label, value, title) {
  return html`<div class="cd-stat" title=${title || ''}>
    <div class="v mono">${value == null || value === '' ? '—' : value}</div>
    <div class="k">${label}</div>
  </div>`;
}
