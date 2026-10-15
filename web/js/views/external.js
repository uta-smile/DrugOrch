/* The public board: DrugOrch beside the published state of the art.
 *
 * The routing board next door answers an internal question, whether the orchestrator beats
 * this project's own baselines. A reader outside the project has no reason to trust a
 * benchmark its author also designed. This board answers the external question instead: a
 * benchmark somebody else built, scored by that benchmark's own evaluator, against the
 * numbers its authors published.
 *
 * Two things are therefore on screen that a leaderboard usually hides.
 *
 * **Which rows were measured here.** A row we ran and a row read out of a paper carry
 * different warranties, and a table that mixes them silently is not evidence. Measured rows
 * are marked, published rows name their source, and the two are never sorted into one
 * undifferentiated ranking without that mark.
 *
 * **What each row was measured on.** Every row carries its instance count. Published rows
 * cover all 1,000 instances of a task; a row measured here covers the sample that was run,
 * and the sample's composition is stated under the table rather than in a footnote nobody
 * opens.
 *
 * The no-tool control sits in the same table as the orchestrated arm on purpose. The
 * interesting quantity is the distance between them, because that distance is the part
 * attributable to the orchestration rather than to the model underneath it.
 */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { MetricTerm } from '@/components/MetricTerm.js';
import { getExternalBench } from '@/api.js';

const TASKS = [
  ['generation', 'Generation', 'Design a molecule that meets every stated and implied constraint.'],
  ['optimization', 'Optimization', 'Edit a given lead so it meets the constraints, preserving what must survive.'],
];

const ARM_LABEL = {
  drugorch: 'DrugOrch (orchestrated)',
  notool: 'DrugOrch model, no tools',
};

export function ExternalBoard() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [task, setTask] = useState('generation');

  useEffect(() => {
    getExternalBench().then(setData).catch((e) => setErr(e.message));
  }, []);

  if (err) return html`<div class="note err">${err}</div>`;
  if (!data) return html`<div class="empty">loading…</div>`;

  const mine = (data.rows || []).filter((r) => r.scope === task);
  const published = (data.published || []).filter((r) => r.scope === task);
  const rows = [...mine, ...published].sort((a, b) => b.success - a.success);
  const src = data.published_source || {};
  const sample = data.sample || {};
  const best = mine.find((r) => r.arm === 'drugorch');
  const bestPublished = published.reduce(
    (acc, r) => (acc == null || r.success > acc.success ? r : acc),
    null,
  );

  return html`
    <div>
      <div class="bench-head">
        <div>
          <h2>MolDesignBench</h2>
          <p class="muted small">
            2,000 scenario-grounded design instances with explicit and implied constraints.
            ${' '}Answers here are scored by the benchmark's own evaluator, in the
            environment its answer key was built in.
          </p>
        </div>
        <span class="spacer"></span>
        ${best &&
        bestPublished &&
        html`<div class="bench-winner">
          <div class="k">success, this task</div>
          <div class="v">
            ${(best.success * 100).toFixed(0)}%
            <span class="small muted" style="font-weight:400">
              vs ${(bestPublished.success * 100).toFixed(0)}% ${bestPublished.arm}
            </span>
          </div>
        </div>`}
      </div>

      <div class="seg" role="tablist">
        ${TASKS.map(
          ([key, label]) => html`<button
            key=${key}
            role="tab"
            aria-selected=${task === key}
            class=${'seg-btn' + (task === key ? ' on' : '')}
            onClick=${() => setTask(key)}
          >
            ${label}
          </button>`,
        )}
        <span class="spacer"></span>
        <span class="small muted">${TASKS.find((t) => t[0] === task)[2]}</span>
      </div>

      ${!data.available &&
      html`<div class="note warn" style="margin:12px 0">
        <b>No run has been scored here yet.</b> The published rows below are the paper's.
        ${' '}Measuring our own row is a command, because it spends model calls:
        <pre class="mono tiny scroller" style="margin-top:8px">${data.hint ||
          'drugorch extbench --n 200'}</pre>
      </div>`}

      <${SuccessChart} rows=${rows} />

      <table class="board">
        <thead>
          <tr>
            <th>arm</th>
            <th>tools</th>
            <th class="num" title="Instances behind the row">n</th>
            <th class="num" id="ext-success"><${MetricTerm} name="success" label="success" /></th>
            <th class="num"><${MetricTerm} name="csr" label="CSR" /></th>
            <th class="num"><${MetricTerm} name="mcd" label="MCD" /></th>
            <th class="num"><${MetricTerm} name="inf_acc" label="InfAcc" /></th>
            <th class="num"><${MetricTerm} name="mcc" label="MCC" /></th>
            <th class="num"><${MetricTerm} name="validity" label="validity" /></th>
            <th class="num"><${MetricTerm} name="tool_calls" label="calls" /></th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(
            (r, i) => html`<tr
              key=${r.arm + r.scope + i}
              class=${r.measured_here ? 'lead' : ''}
            >
              <td>
                <b>${r.measured_here ? ARM_LABEL[r.arm] || r.arm : r.arm}</b>
                ${r.measured_here
                  ? html`<span class="sd">measured here</span>`
                  : html`<span class="sd">${r.source || 'published'}</span>`}
              </td>
              <td class="small muted">${r.tools ? 'yes' : 'no'}</td>
              <td class="num mono">${r.n}</td>
              <td class="num mono"><b>${r.success.toFixed(3)}</b></td>
              <td class="num mono">${r.csr.toFixed(3)}</td>
              <td class="num mono">${r.mcd.toFixed(2)}</td>
              <td class="num mono">${r.inf_acc.toFixed(3)}</td>
              <td class="num mono">${r.mcc.toFixed(3)}</td>
              <td class="num mono">${r.validity.toFixed(3)}</td>
              ${/* A no-tool arm has no call count, rather than a count of zero. Printing
                    0.00 invites it to be read as an arm that had tools and declined them. */ ''}
              <td class="num mono">
                ${!r.tools || r.tool_calls == null ? '—' : r.tool_calls.toFixed(2)}
              </td>
            </tr>`,
          )}
        </tbody>
      </table>

      <div class="ext-notes">
        ${/* The toolsets are a page of their own and a link from here, because a score is
              not interpretable without them: seventeen tools against four is most of what
              separates these rows. */ ''}
        <div class="note">
          <b>The two sides are not equipped alike.</b> The benchmark gives its models 17
          ${' '}tools; DrugOrch answers with 4 subsystems.
          ${' '}<a href="#/tools">What each side has, and what each tool does</a>.
        </div>
        <div class="note">
          <b>Highlighted rows were measured here.</b> The rest are
          ${' '}${src.paper ? `${src.paper}, ${src.table}, ${src.venue}` : 'published figures'},
          ${' '}${src.n_per_task || 1000} instances per task. They were not re-run.
        </div>
        ${data.available &&
        html`<div class="note">
          <b>The sample is stated rather than assumed.</b> ${sample.n} instances at
          ${' '}fingerprint <span class="mono">${sample.fingerprint}</span>:
          ${' '}${sample.by_task?.generation} generation, ${sample.by_task?.optimization}
          ${' '}optimization, ${sample.infeasible} of them contradictory by construction,
          ${' '}${sample.mean_constraints} constraints each on average.
        </div>`}
        ${data.available &&
        html`<div class="note">
          <b>Measurement is the benchmark's.</b> Properties come from its own analyzer and
          ${' '}functional groups from its own counter, both run in
          ${' '}<span class="mono">${data.provenance?.measured_in}</span>. Scoring runs
          ${' '}<span class="mono">${data.provenance?.scored_by}</span>.
        </div>`}
        ${fallbackNote(data)}
      </div>

      <${PairedGain} data=${data} task=${task} />
      <${Decomposition} data=${data} />

      ${data.available && html`<${ArmDetail} data=${data} />`}
      <${Citation} src=${src} />
    </div>
  `;
}

/* The citation, on the page rather than in a repository file.
 *
 * Every published row here is somebody else's measurement, and the people who made it are
 * named where the numbers are shown. The BibTeX is included because the next thing anyone
 * doing this seriously needs is a citation they can paste, and retyping one from a rendered
 * table is how author lists lose their last two names.
 *
 * The link goes to the benchmark rather than to a copy of its results: a reader checking
 * whether our row is fair needs the instances and the evaluator, not a prettier table.
 */
function Citation({ src }) {
  const [copied, setCopied] = useState(false);
  if (!src?.paper) return null;

  const authors = (src.authors || []).join('; ');
  const copy = () => {
    navigator.clipboard?.writeText(src.bibtex || '').then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1800);
      },
      () => setCopied(false),
    );
  };

  return html`<section class="cite">
    <h3>The benchmark</h3>
    <p class="ref">
      ${authors}. <b>${src.paper}</b>. ${src.venue}, ${src.year}.
      ${' '}Published results quoted here are ${src.table}.
    </p>
    <p class="small muted">
      ${src.licence}
      ${' '}Rows above were read from
      ${' '}<span class="mono tiny">${src.file}</span> and are checked against it by the
      ${' '}test suite.
    </p>
    <div class="cite-links">
      ${src.code_url &&
      html`<a href=${src.code_url} target="_blank" rel="noreferrer noopener">Benchmark code</a>`}
      ${src.data_url &&
      html`<a href=${src.data_url} target="_blank" rel="noreferrer noopener">Dataset</a>`}
      ${src.bibtex &&
      html`<button class="btn ghost tiny" onClick=${copy}>
        ${copied ? 'copied' : 'copy BibTeX'}
      </button>`}
    </div>
    ${src.bibtex &&
    html`<details>
      <summary class="small muted">BibTeX</summary>
      <pre class="mono tiny scroller">${src.bibtex}</pre>
    </details>`}
  </section>`;
}

/* Success as bars, above the table that carries every other column.
 *
 * One quantity, ordered, across about a dozen rows: the job is magnitude comparison, so bars
 * on a shared baseline. The table below is the data view, which is why only three bars are
 * labelled with their value (ours, and the leading published row) rather than all of them.
 *
 * Two fills, not twelve. Colour here encodes provenance, which is the distinction a reader
 * must not miss: measured here, or read from the paper. Rows measured here take the
 * interface's accent; published rows take a recessive neutral. Provenance is also in the row
 * label and in the legend, so it is never carried by colour alone. The accent is the one the
 * rest of the interface already uses, in both themes, so a highlighted bar reads as the same
 * kind of thing as every other highlight on the page.
 */
function SuccessChart({ rows }) {
  if (!rows.length) return null;

  const H = 22; // one row
  const GAP = 3; // a surface gap between adjacent bars, so fills never touch
  const LABEL = 190;
  const W = 760;
  const plot = W - LABEL - 54;
  const max = Math.max(0.5, ...rows.map((r) => r.success));
  const height = rows.length * (H + GAP);
  const leader = rows.find((r) => !r.measured_here);

  return html`<figure class="ext-chart">
    <figcaption>
      Success rate, this task
      <span class="legend">
        <span class="key"><i class="swatch mine"></i>measured here</span>
        <span class="key"><i class="swatch pub"></i>published</span>
      </span>
    </figcaption>
    <svg viewBox=${`0 0 ${W} ${height}`} width="100%" height=${height} role="img"
         aria-label="Success rate by arm, measured rows and published rows">
      ${[0, 0.25, 0.5].filter((t) => t <= max).map(
        (t) => html`<line key=${'g' + t} class="grid"
          x1=${LABEL + (t / max) * plot} x2=${LABEL + (t / max) * plot}
          y1="0" y2=${height - GAP} />`,
      )}
      ${rows.map((r, i) => {
        const y = i * (H + GAP);
        const w = Math.max(2, (r.success / max) * plot);
        const label = r.measured_here || r === leader;
        return html`<g key=${r.arm + i} class=${r.measured_here ? 'bar mine' : 'bar'}>
          <title>${`${r.arm}${r.tools ? ', with tools' : ', no tools'}: ${(r.success * 100).toFixed(1)}% of ${r.n} instances`}</title>
          ${/* Our own labels already say whether the arm had tools, so the suffix is only
                for published rows, where the same model appears twice. */ ''}
          <text class="cat" x=${LABEL - 8} y=${y + H / 2 + 4} text-anchor="end">
            ${r.measured_here
              ? ARM_LABEL[r.arm] || r.arm
              : r.arm + (r.tools ? '' : ' (no tools)')}
          </text>
          <rect class="hit" x=${LABEL} y=${y} width=${plot} height=${H} />
          <rect class="fill" x=${LABEL} y=${y + 2} width=${w} height=${H - 4} rx="4" />
          ${label &&
          html`<text class="val" x=${LABEL + w + 7} y=${y + H / 2 + 4}>
            ${(r.success * 100).toFixed(0)}%
          </text>`}
        </g>`;
      })}
    </svg>
  </figure>`;
}

/* Why the score is what it is, which no column on the board can say.
 *
 * Failing an instance here has two causes that call for opposite fixes: the arm worked from
 * an incomplete reading of the question, or the reading was right and the chemistry did not
 * get there. A success rate cannot separate them. This can, because the arm writes down the
 * constraint set it believes it was asked for, and that set can be compared with the key
 * once scoring is finished.
 *
 * The comparison is made after the run, never during it. The arm does not see the key at
 * any point, which is the property `test_no_prompt_can_carry_the_answer_key` exists to
 * hold. A diagnostic derived from the key describes a run that has already happened.
 */
function Decomposition({ data }) {
  const d = data.diagnostics?.drugorch;
  if (!d || !d.n_feasible) return null;

  const pct = (x) => `${(x * 100).toFixed(0)}%`;
  return html`<div class="card" style="margin-top:18px">
    <h3>
      Where the score goes
      <span class="spacer"></span>
      <span class="small muted" style="font-weight:400;text-transform:none">
        reading the question, against solving what was read
      </span>
    </h3>
    <div class="body">
      <div class="decomp">
        <div class="stat">
          <div class="k">specification recall</div>
          <div class="v">${pct(d.spec_recall)}</div>
          <div class="h">
            ${d.constraints_extracted} constraints written down of the
            ${' '}${d.constraints_in_key} the key holds, averaged over
            ${' '}${d.n_feasible} feasible instances. Under-extracted on
            ${' '}${d.under_extracted}.
          </div>
        </div>
        <div class="stat">
          <div class="k">confidence precision</div>
          <div class="v">${pct(d.confidence_precision)}</div>
          <div class="h">
            How often "this satisfies the specification" was also true of the key.
            ${' '}${d.believed_and_wrong} instances were finished against a reading that
            ${' '}turned out incomplete.
          </div>
        </div>
        <div class="stat">
          <div class="k">verifications spent</div>
          <div class="v">${d.verifications}</div>
          <div class="h">
            Re-readings of the question before answering. Each is one request and no
            ${' '}subsystem call.
          </div>
        </div>
        <div class="stat">
          <div class="k">calls, solved against failed</div>
          <div class="v">${d.mean_calls_when_solved} / ${d.mean_calls_when_failed}</div>
          <div class="h">
            Subsystem calls per instance. Fewer on the instances it solved means the budget
            ${' '}is not what limits the score.
          </div>
        </div>
      </div>
    </div>
  </div>`;
}

/* What the orchestration is worth, as opposed to what the model underneath it is worth.
 *
 * The table ranks arms. It cannot say whether a difference between two of our own rows is
 * real, because at a hundred instances and success rates in the tens, the instances an arm
 * happened to solve carry a lot of chance. The two arms answered the same questions, so the
 * comparison is made per instance: how many did each solve that the other did not.
 *
 * Instances both solved, and instances neither solved, say nothing about which arm is
 * better and are reported separately rather than folded in. The p-value is an exact
 * binomial over the instances where the arms disagreed.
 *
 * It is stated whether or not it favours the orchestration. A result that is only shown
 * when it is positive is not a measurement.
 */
function PairedGain({ data, task }) {
  const g = (data.paired || []).find((p) => p.scope === task);
  if (!g || !g.n_paired) return null;

  const decisive = g.p_value < 0.05;
  const better = g.delta > 0;
  const verdict = !decisive
    ? 'The two are not distinguishable at this sample size.'
    : better
      ? 'The orchestration accounts for the difference, not the model underneath it.'
      : 'The no-tool control is ahead here, and the orchestration is not paying for itself.';

  return html`<div class=${'note paired ' + (decisive ? (better ? 'ok' : 'warn') : '')}>
    <b>Orchestrated against the same model with no tools, paired on
    ${' '}${g.n_paired} instances.</b>
    ${' '}The orchestrated arm solved <b>${g.arm_only}</b> that the control did not; the
    ${' '}control solved <b>${g.baseline_only}</b> that it did not.
    ${' '}Both solved ${g.both}; neither solved ${g.neither}.
    ${' '}Exact McNemar <span class="mono">p = ${g.p_value < 0.001
      ? g.p_value.toExponential(1)
      : g.p_value.toFixed(3)}</span>. ${verdict}
  </div>`;
}

/* A row with fallbacks is not a row for that arm: its decisions never reached the model, so
 * what it measured was the absence of the arm. Shown with the table rather than under it. */
function fallbackNote(data) {
  const bad = Object.entries(data.arms || {}).filter(([, a]) => a.n_fallbacks > 0);
  if (!bad.length) return null;
  return html`<div class="note warn">
    ${bad.map(
      ([name, a]) => html`<div key=${name}>
        <b>${name}</b>: ${a.n_fallbacks} decision(s) never reached the model. Those
        instances are not results for this arm.
      </div>`,
    )}
  </div>`;
}

/* What the orchestrated arm actually did. A success rate with no account of the work behind
 * it cannot be read as evidence about orchestration: an arm that never called a subsystem
 * and an arm that called four per instance are different systems with the same column. */
function ArmDetail({ data }) {
  const arms = Object.entries(data.arms || {});
  if (!arms.length) return null;
  return html`<div class="card" style="margin-top:18px">
    <h3>
      What each arm did
      <span class="spacer"></span>
      <span class="small muted" style="font-weight:400;text-transform:none">
        the work behind the score, not the score
      </span>
    </h3>
    <div class="body flush">
      <table class="board">
        <thead>
          <tr>
            <th>arm</th>
            <th>model</th>
            <th class="num">instances</th>
            <th class="num">mean calls</th>
            <th class="num">abstained</th>
            <th class="num">mean seconds</th>
            <th>subsystems used</th>
          </tr>
        </thead>
        <tbody>
          ${arms.map(
            ([name, a]) => html`<tr key=${name}>
              <td><b>${ARM_LABEL[name] || name}</b></td>
              ${/* Model, route and effort together. Any one of them changes what the row
                    measured, and a row that names only the model is not reproducible. */ ''}
              <td class="mono tiny">
                ${a.backend?.model || '—'}
                <span class="sd">
                  ${a.backend?.route || a.backend?.backend || ''}
                  ${a.backend?.effort ? `, effort ${a.backend.effort}` : ''}
                </span>
              </td>
              <td class="num mono">${a.n_attempts}</td>
              <td class="num mono">${a.mean_tool_calls}</td>
              <td class="num mono">${a.abstained}</td>
              <td class="num mono">${a.mean_wall_s}</td>
              <td class="small">
                ${Object.entries(a.tool_usage || {})
                  .map(([t, n]) => `${t} ${n}`)
                  .join(', ') || '—'}
              </td>
            </tr>`,
          )}
        </tbody>
      </table>
    </div>
  </div>`;
}
