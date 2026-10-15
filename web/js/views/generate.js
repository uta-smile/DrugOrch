/* The generation workbench: the operations, without the experiment around them.
 *
 * Every other page here is about a *run* — a budgeted, routed campaign whose point is to be
 * compared against another campaign. That shape answers this project's research question and
 * is the wrong shape for someone who simply wants to see a molecule in 3D, elaborate a
 * scaffold, or find out which of forty analogues fits a pocket. Making them start a run to
 * get one conformer is an experimental protocol leaking into a tool.
 *
 * Three panels, in the order the work actually goes:
 *
 *   build      a SMILES has no geometry; this supplies it. Exact and instant.
 *   elaborate  one or two known molecules in, analogues out.
 *   fit        a real pocket scores them, and the pose can be looked at.
 *
 * They chain through one shared basket, because the useful motion is elaborate → fit →
 * look at the winner, and a workbench that made you copy SMILES between three pages would
 * not get used for that.
 *
 * On the direction nobody can do here honestly: *generating a molecule from a pocket* with a
 * learned prior — Pocket2Mol, TargetDiff, DiffSBDD — needs a model that is not installed and
 * has not been verified to run on this machine. The "fit" panel is structure-conditioned
 * *selection*, not generation: chemistry proposes, the pocket disposes. That is weaker than
 * a learned prior and completely honest about what the numbers mean, which is the trade this
 * project makes everywhere else too.
 */

import { useEffect, useRef, useState } from 'preact/hooks';
import { html } from '@/h.js';
import {
  depictUrl,
  genDock,
  genElaborate,
  genEmbed,
  genJob,
  genJobCancel,
  genPocket,
} from '@/api.js';
import { Mol3D, downloadSdf } from '@/components/Mol3D.js';

const EXAMPLES = {
  ruxolitinib: 'N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1',
  imatinib:
    'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1',
  aspirin: 'CC(=O)Oc1ccccc1C(=O)O',
};

const MODES = [
  ['build', '3D structure', 'A SMILES has no geometry. Build one.'],
  ['elaborate', 'Analogues', 'Grow variations on a molecule you already have.'],
  ['fit', 'Fit a pocket', 'Score molecules against a real binding site.'],
];

export function GenerateView() {
  const [mode, setMode] = useState('build');
  // The basket is what makes this one workbench rather than three tools. Elaborating fills
  // it; the pocket panel reads it.
  const [basket, setBasket] = useState([]);

  return html`
    <div class="wb">
      <div class="wb-head">
        <div>
          <h2>Workbench</h2>
          <p class="small muted">
            Build conformers, generate analogues, or dock SMILES against a protein pocket.
            Workbench operations are not saved as runs.
          </p>
        </div>
        <span class="spacer"></span>
        ${basket.length > 0 &&
        html`<span class="pill" title=${basket.join('\n')}>
          ${basket.length} molecule${basket.length === 1 ? '' : 's'} carried over
        </span>`}
      </div>

      <div class="wb-modes">
        ${MODES.map(
          ([k, label, blurb]) => html`
            <button type="button" key=${k} class=${'wb-mode' + (mode === k ? ' sel' : '')}
                    onClick=${() => setMode(k)}>
              <b>${label}</b>
              <span class="small muted">${blurb}</span>
            </button>
          `,
        )}
      </div>

      ${mode === 'build' && html`<${BuildPanel} basket=${basket} />`}
      ${mode === 'elaborate' &&
      html`<${ElaboratePanel} onBasket=${setBasket} onDock=${() => setMode('fit')} />`}
      ${mode === 'fit' && html`<${FitPanel} basket=${basket} onBasket=${setBasket} />`}
    </div>
  `;
}

/* --- build: SMILES -> 3D ---------------------------------------------------- */

function BuildPanel({ basket }) {
  const [text, setText] = useState(basket.length ? basket.join('\n') : EXAMPLES.ruxolitinib);
  const [data, setData] = useState(null);
  const [sel, setSel] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  async function build(e) {
    e?.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const smiles = text.split('\n').map((s) => s.trim()).filter(Boolean);
      const d = await genEmbed(smiles);
      setData(d);
      setSel(0);
    } catch (ex) {
      setErr(ex.message);
    } finally {
      setBusy(false);
    }
  }

  const current = data?.structures?.[sel] || null;

  return html`
    <div class="wb-grid">
      <form class="card" onSubmit=${build}>
        <h3>Molecules</h3>
        <div class="body">
          <textarea rows="7" class="mono" value=${text}
                    onInput=${(e) => setText(e.target.value)}
                    placeholder="One SMILES per line"></textarea>
          <div class="row" style="gap:7px;margin-top:8px;flex-wrap:wrap">
            ${Object.entries(EXAMPLES).map(
              ([name, smi]) => html`<button type="button" key=${name} class="btn ghost small"
                      onClick=${() => setText(smi)}>${name}</button>`,
            )}
            <span class="spacer"></span>
            <button class="btn primary" disabled=${busy}>${busy ? 'Building…' : 'Build 3D'}</button>
          </div>
          ${err && html`<p class="note err">${err}</p>`}
          <p class="small muted" style="margin-top:10px">
            Conformers use ETKDGv3 distance geometry and crystal-derived torsion preferences,
            then MMFF94 relaxation. Coordinates are deterministic for each input molecule.
          </p>
        </div>
      </form>

      <div class="card">
        <h3>
          Structure
          <span class="spacer"></span>
          ${current &&
          html`<button class="btn ghost small"
                       onClick=${() => downloadSdf(current.sdf, current.smiles)}>
            download .sdf
          </button>`}
        </h3>
        <div class="body flush">
          ${!data && html`<div class="empty">Build a molecule to see its geometry.</div>`}
          ${current &&
          html`
            <${Mol3D} sdf=${current.sdf} height=${300} style="ballstick" label="conformer" />
            <div class="wb-stats">
              ${stat('heavy atoms', current.n_heavy_atoms)}
              ${stat('rotatable', current.n_rotatable_bonds,
                     'Torsions. A flexible molecule has many low-energy shapes, so one conformer says less about it.')}
              ${stat('MMFF kcal/mol', current.mmff_energy,
                     'Energy of this relaxed conformer. Comparable between conformers of the SAME molecule only.')}
              ${stat('QED', current.panel?.scores?.qed)}
              ${stat('SA', current.panel?.scores?.sa_score, 'Synthetic accessibility, 1–10, lower is easier.')}
            </div>
            ${current.n_rotatable_bonds > 8 &&
            html`<p class="cd-warn" style="margin:8px">
              ${current.n_rotatable_bonds} rotatable bonds. This compound has many accessible
              shapes, and the one shown is a low-energy guess rather than the conformation
              it would adopt in a pocket. Dock it if that distinction matters.
            </p>`}
          `}
          ${data?.structures?.length > 1 &&
          html`<div class="pose-list">
            ${data.structures.map(
              (s, i) => html`<button key=${s.smiles} class=${'pose-item' + (i === sel ? ' on' : '')}
                      onClick=${() => setSel(i)} title=${s.smiles}>
                <span class="score">${s.n_heavy_atoms}</span>
                <span class="smi">${s.smiles}</span>
              </button>`,
            )}
          </div>`}
          ${data?.failed?.length > 0 &&
          html`<div class="small" style="padding:8px;color:var(--warn)">
            ${data.failed.map(
              (f) => html`<div key=${f.smiles}><span class="mono">${f.smiles}</span>: ${f.error}</div>`,
            )}
          </div>`}
        </div>
      </div>
    </div>
  `;
}

/* --- elaborate: seed -> analogues ------------------------------------------- */

function ElaboratePanel({ onBasket, onDock }) {
  const [seeds, setSeeds] = useState(EXAMPLES.ruxolitinib);
  const [n, setN] = useState(16);
  const [seed, setSeed] = useState(42);
  const [rate, setRate] = useState(0.5);
  const [data, setData] = useState(null);
  const [picked, setPicked] = useState(() => new Set());
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  async function run(e) {
    e?.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const d = await genElaborate({
        seeds: seeds.split('\n').map((s) => s.trim()).filter(Boolean),
        n: Number(n),
        seed: Number(seed),
        mutation_rate: Number(rate),
      });
      setData(d);
      setPicked(new Set(d.children.map((c) => c.smiles)));
    } catch (ex) {
      setErr(ex.message);
    } finally {
      setBusy(false);
    }
  }

  function toggle(smi) {
    const next = new Set(picked);
    if (next.has(smi)) next.delete(smi);
    else next.add(smi);
    setPicked(next);
  }

  const children = data?.children || [];

  return html`
    <div>
      <form class="card" onSubmit=${run}>
        <h3>Seeds</h3>
        <div class="body">
          <textarea rows="3" class="mono" value=${seeds}
                    onInput=${(e) => setSeeds(e.target.value)}
                    placeholder="One SMILES per line. Two or more enables crossover."></textarea>
          <div class="row" style="gap:14px;margin-top:9px;flex-wrap:wrap;align-items:flex-end">
            ${num('How many', n, setN, { min: 1, max: 64 })}
            ${num('Seed', seed, setSeed, {})}
            ${num('Mutation rate', rate, setRate, { min: 0, max: 1, step: 0.1 })}
            <span class="spacer"></span>
            <button class="btn primary" disabled=${busy}>
              ${busy ? 'Generating…' : 'Generate analogues'}
            </button>
          </div>
          ${err && html`<p class="note err">${err}</p>`}
          <p class="small muted" style="margin-top:10px">
            GraphGA creates analogues by crossover between two seeds or by single-atom
            mutation. One seed enables mutation only. The random seed makes runs reproducible.
          </p>
        </div>
      </form>

      ${data &&
      html`
        <div class="card" style="margin-top:14px">
          <h3>
            Analogues
            <span class="spacer"></span>
            <span class="small muted" style="font-weight:400;text-transform:none">
              ${picked.size} of ${children.length} selected
            </span>
          </h3>
          <div class="body">
            <p class="cd-why">${data.note}.</p>
            <div class="cd-chips" style="margin-bottom:10px">
              ${Object.entries(data.operators || {}).map(
                ([k, v]) => html`<span class="chip ok" key=${k}>${v} ${k}</span>`,
              )}
              ${Object.entries(data.rejected || {}).map(
                ([k, v]) => html`<span class="chip" key=${k}
                  title=${REJECT_WHY[k] || 'attempts that produced nothing usable'}>${v} ${k}</span>`,
              )}
              ${data.yield != null &&
              html`<span class="chip" title="children per attempt; a low yield means the seeds are nearly exhausted">
                yield ${(data.yield * 100).toFixed(0)}%
              </span>`}
            </div>

            <div class="wb-mols">
              ${children.map(
                (c) => html`
                  <button type="button" key=${c.smiles}
                          class=${'wb-mol' + (picked.has(c.smiles) ? ' on' : '')}
                          onClick=${() => toggle(c.smiles)} title=${c.smiles}>
                    <img src=${depictUrl(c.smiles, 200, 150)} alt="" loading="lazy"
                         width="120" height="90" />
                    <div class="tiny muted">${c.operator}</div>
                    <div class="tiny mono">
                      QED ${fmt(c.panel?.scores?.qed)} · SA ${fmt(c.panel?.scores?.sa_score)}
                    </div>
                    ${c.panel?.flags?.pains_alert &&
                    html`<div class="tiny" style="color:var(--warn)">PAINS</div>`}
                  </button>
                `,
              )}
            </div>

            <div class="row" style="gap:9px;margin-top:12px">
              <button type="button" class="btn ghost small"
                      onClick=${() => setPicked(new Set(children.map((c) => c.smiles)))}>
                select all
              </button>
              <button type="button" class="btn ghost small" onClick=${() => setPicked(new Set())}>
                clear
              </button>
              <span class="spacer"></span>
              <button type="button" class="btn primary" disabled=${picked.size === 0}
                      onClick=${() => { onBasket([...picked]); onDock(); }}>
                Score ${picked.size} against a pocket →
              </button>
            </div>
          </div>
        </div>
      `}
    </div>
  `;
}

const REJECT_WHY = {
  duplicate: 'Already produced, or identical to a seed. Mostly duplicates means the seeds are exhausted.',
  mutation_failed: 'The perturbation produced nothing that survived sanitization or the size filters.',
  crossover_failed:
    'No valid child from recombination. Crossover cuts at acyclic single bonds, and small rigid molecules offer few.',
};

/* --- fit: molecules -> a pocket --------------------------------------------- */

function FitPanel({ basket, onBasket }) {
  const [pdb, setPdb] = useState('7LL4');
  const [lig, setLig] = useState('');
  const [pocket, setPocket] = useState(null);
  const [text, setText] = useState(basket.join('\n') || EXAMPLES.ruxolitinib);
  const [job, setJob] = useState(null);
  const [rows, setRows] = useState(null);
  const [sel, setSel] = useState(0);
  const [receptor, setReceptor] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => {
    if (basket.length) setText(basket.join('\n'));
  }, [basket.join('\n')]);

  // Poll while a dock job runs. Docking is tens of seconds per molecule, so the page has to
  // say which molecule rather than show a spinner of unknown duration.
  useEffect(() => {
    if (!job || job.status !== 'running') return undefined;
    pollRef.current = setInterval(async () => {
      try {
        const j = await genJob(job.job_id);
        setJob(j);
        if (j.status === 'done') {
          setRows(j.result.rows);
          setSel(0);
        }
        if (j.status !== 'running') clearInterval(pollRef.current);
      } catch (e) {
        setErr(e.message);
        clearInterval(pollRef.current);
      }
    }, 1200);
    return () => clearInterval(pollRef.current);
  }, [job?.job_id, job?.status]);

  async function prepare(e) {
    e?.preventDefault();
    setBusy(true);
    setErr(null);
    setPocket(null);
    setRows(null);
    try {
      const p = await genPocket(pdb.trim(), lig.trim());
      setPocket(p);
      // The receptor is fetched once, here, rather than per pose: it is the expensive
      // geometry and rebuilding it on every selection froze the renderer.
      const text3d = await fetch(`/api/gen/receptor?pdb_id=${encodeURIComponent(pdb.trim())}` +
                                 `&ligand_code=${encodeURIComponent(lig.trim())}`)
        .then((r) => (r.ok ? r.text() : null))
        .catch(() => null);
      setReceptor(text3d);
    } catch (ex) {
      setErr(ex.message);
    } finally {
      setBusy(false);
    }
  }

  async function dock() {
    setErr(null);
    setRows(null);
    try {
      const smiles = text.split('\n').map((s) => s.trim()).filter(Boolean);
      const { job_id } = await genDock({
        pdb_id: pdb.trim(), ligand_code: lig.trim(), smiles,
      });
      setJob({ job_id, status: 'running', done: 0, total: smiles.length });
    } catch (ex) {
      setErr(ex.message);
    }
  }

  const current = rows?.[sel] || null;
  const scored = (rows || []).filter((r) => r.score != null);

  return html`
    <div>
      <form class="card" onSubmit=${prepare}>
        <h3>Pocket</h3>
        <div class="body">
          <div class="row" style="gap:14px;align-items:flex-end;flex-wrap:wrap">
            ${txt('PDB id', pdb, setPdb, '7LL4')}
            ${txt('Ligand code', lig, setLig, 'auto')}
            <button class="btn" disabled=${busy}>${busy ? 'Preparing…' : 'Prepare pocket'}</button>
            ${pocket &&
            html`<span class="pill done">${pocket.pdb_id} · ${pocket.ligand_code} · chain ${pocket.chain}</span>`}
          </div>
          ${pocket &&
          html`<div class="small muted" style="margin-top:9px">
            ${pocket.why}. ${pocket.n_residues} residues; search box${' '}
            ${pocket.box.size_x.toFixed(0)}×${pocket.box.size_y.toFixed(0)}×${pocket.box.size_z.toFixed(0)} Å${' '}
            centred on the bound ligand.
            ${!pocket.vina &&
            html`<span style="color:var(--err)">
              No Vina executable found, so nothing can be scored against this pocket.
            </span>`}
          </div>`}
          ${err && html`<p class="note err">${err}</p>`}
          <p class="small muted" style="margin-top:10px">
            Pocket preparation uses a ligand-bound structure to locate the binding site.
            Structures without a bound ligand cannot define the docking box.
          </p>
        </div>
      </form>

      <div class="card" style="margin-top:14px">
        <h3>Molecules to score</h3>
        <div class="body">
          <textarea rows="5" class="mono" value=${text}
                    onInput=${(e) => setText(e.target.value)}
                    placeholder="One SMILES per line"></textarea>
          <div class="row" style="gap:9px;margin-top:8px">
            ${basket.length > 0 &&
            html`<button type="button" class="btn ghost small" onClick=${() => onBasket([])}>
              forget carried-over molecules
            </button>`}
            <span class="spacer"></span>
            <button type="button" class="btn primary"
                    disabled=${!pocket || !pocket.vina || job?.status === 'running'}
                    onClick=${dock}>
              ${job?.status === 'running' ? 'Docking…' : 'Dock into pocket'}
            </button>
          </div>
          ${!pocket &&
          html`<p class="small muted" style="margin-top:8px">Prepare a pocket first.</p>`}

          ${job?.status === 'running' &&
          html`<div style="margin-top:11px">
            <div class="row small muted" style="justify-content:space-between">
              <span>${job.message || 'starting…'} · ${job.elapsed_s ?? 0}s</span>
              <button type="button" class="btn ghost small"
                      onClick=${() => genJobCancel(job.job_id).catch(() => {})}>stop</button>
            </div>
            <div class="meter" style="margin-top:4px">
              <i style=${`width:${job.total ? (job.done / job.total) * 100 : 0}%`}></i>
            </div>
          </div>`}
          ${job?.status === 'failed' && html`<p class="note err">${job.error}</p>`}
        </div>
      </div>

      ${rows &&
      html`
        <div class="wb-grid" style="margin-top:14px">
          <div class="card">
            <h3>Ranked <span class="spacer"></span>
              <span class="small muted" style="font-weight:400;text-transform:none">
                ${scored.length} of ${rows.length} docked
              </span>
            </h3>
            <div class="body flush">
              <table class="cd-table" style="padding:0 10px">
                <thead><tr><th>kcal/mol</th><th>QED</th><th>molecule</th></tr></thead>
                <tbody>
                  ${rows.map(
                    (r, i) => html`<tr key=${r.smiles} class=${i === sel ? 'cur' : ''}
                            style="cursor:pointer" onClick=${() => setSel(i)}>
                      <td class="num mono">
                        ${r.score == null ? '—' : (-r.score).toFixed(2)}
                      </td>
                      <td class="num mono">${fmt(r.panel?.scores?.qed)}</td>
                      <td class="smi mono" title=${r.note}>${r.smiles}</td>
                    </tr>`,
                  )}
                </tbody>
              </table>
              <p class="small muted" style="padding:8px 10px 0">
                Vina scores rank poses; they are not measured binding affinities.
              </p>
            </div>
          </div>

          <div class="card">
            <h3>
              Pose
              <span class="spacer"></span>
              ${current?.score != null &&
              html`<a class="btn ghost small"
                      href=${`/api/gen/jobs/${encodeURIComponent(job.job_id)}/pose/${sel}`}
                      download=${`pose_${sel + 1}.sdf`}>download pose</a>`}
            </h3>
            <div class="body flush">
              ${current?.score == null &&
              html`<div class="empty">${current?.note || 'This molecule did not dock.'}</div>`}
              ${/* Deliberately no key on the selection. Keying it per row would remount the
                    viewer on every click, tearing down the WebGL context and rebuilding the
                    2,400-atom receptor cartoon each time — the exact thing that froze the
                    run viewer. Updating in place swaps only the ligand. */ ''}
              ${current?.score != null &&
              html`<${PoseFromJob} jobId=${job.job_id} index=${sel} receptor=${receptor} />`}
              ${current?.score != null &&
              html`<div class="wb-stats">
                ${stat('kcal/mol', (-current.score).toFixed(2), 'Vina affinity. Ranking, not a measurement.')}
                ${stat('QED', fmt(current.panel?.scores?.qed))}
                ${stat('SA', fmt(current.panel?.scores?.sa_score))}
                ${stat('Ro5', current.panel?.flags?.lipinski_violations)}
              </div>`}
            </div>
          </div>
        </div>
      `}
    </div>
  `;
}

/* Vina writes PDBQT, which 3Dmol misparses silently — it transposes models and atoms and
   drops whole functional groups without raising. The server converts with Meeko, exactly as
   the run viewer's poses are converted, and the browser never sees the raw format. */
function PoseFromJob({ jobId, index, receptor }) {
  const [sdf, setSdf] = useState(null);
  const [err, setErr] = useState(null);

  // The previous pose stays on screen while the next one loads. Clearing it first would
  // unmount the viewer between every click, which rebuilds the receptor cartoon and throws
  // away the WebGL context — and makes the panel flash grey on each selection.
  useEffect(() => {
    let alive = true;
    setErr(null);
    fetch(`/api/gen/jobs/${encodeURIComponent(jobId)}/pose/${index}`)
      .then((r) =>
        r.ok ? r.text() : r.json().then((b) => Promise.reject(new Error(b.detail || r.statusText))),
      )
      .then((t) => alive && setSdf(t))
      .catch((e) => alive && setErr(e.message));
    return () => { alive = false; };
  }, [jobId, index]);

  if (err) return html`<div class="empty">${err}</div>`;
  if (!sdf) return html`<div class="empty">loading pose…</div>`;
  return html`<${Mol3D} sdf=${sdf} receptor=${receptor} height=${320} label="pose" />`;
}

/* --- small helpers ---------------------------------------------------------- */

function stat(label, value, title) {
  return html`<div class="cd-stat" title=${title || ''}>
    <div class="v mono">${value == null ? '—' : value}</div>
    <div class="k">${label}</div>
  </div>`;
}

function num(label, value, set, { min, max, step }) {
  return html`<label class="wb-num">
    <span>${label}</span>
    <input type="number" value=${value} min=${min} max=${max} step=${step ?? 1}
           onInput=${(e) => set(e.target.value)} />
  </label>`;
}

function txt(label, value, set, placeholder) {
  return html`<label class="wb-num">
    <span>${label}</span>
    <input type="text" value=${value} placeholder=${placeholder}
           onInput=${(e) => set(e.target.value)} style="width:110px" />
  </label>`;
}

function fmt(v) {
  return typeof v === 'number' ? v.toFixed(2) : '—';
}
