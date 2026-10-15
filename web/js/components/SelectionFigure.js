/* Why a budget exists when the compute is free.
 *
 * Three curves against pool size: the score a screen would *report*, the true value of the
 * molecule it actually picked, and the best molecule that was in the pool at all. The first
 * runs away, the second crawls, and the gap between them is budget spent discovering the
 * scoring function's error rather than chemistry.
 *
 * Drawn as inline SVG rather than with uPlot, which this project already loads. uPlot is
 * canvas, so it cannot read CSS custom properties and every themed chart here needs a
 * `themechange` listener and a rebuild to follow the light/dark toggle. An SVG element is in
 * the DOM and inherits `var(--ok)` like anything else -- for five points and three series
 * that is the whole of the theme handling, and there is no instance to tear down.
 *
 * The numbers come from the server (`/api/selection-curve`) rather than being computed here.
 * The arithmetic is the argument, and the Python version is the one with tests against it;
 * a second implementation in JavaScript would be an unchecked copy free to drift.
 */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { getSelectionCurve } from '@/api.js';

/* The oracle's error, in kcal/mol. 2.0 is roughly Vina's published affinity RMSE and is the
 * default; 0 is the control, and it is first in the list because being able to watch the
 * whole argument collapse is the point of making this adjustable at all. */
const NOISE_STEPS = [
  { v: 0, label: '0', hint: 'No scoring error' },
  { v: 0.5, label: '0.5', hint: 'Low scoring error' },
  { v: 1, label: '1', hint: 'Moderate scoring error' },
  { v: 2, label: '2', hint: "Approximately AutoDock Vina's published error" },
  { v: 3, label: '3', hint: 'High scoring error' },
];

const W = 660;
const H = 300;
const PAD = { top: 18, right: 14, bottom: 38, left: 52 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

const SERIES = [
  {
    key: 'best_measured',
    cls: 'reported',
    name: 'best score reported',
    note: 'the value a screen would report',
  },
  {
    key: 'true_of_pick',
    cls: 'truth',
    name: 'true value of the pick',
    note: 'measured value of the selected compound',
  },
  {
    key: 'best_true_available',
    cls: 'ceiling',
    name: 'best compound in the pool',
    note: 'best compound available in the pool',
  },
];

function fmtN(n) {
  if (n >= 1_000_000) return `${n / 1_000_000}M`;
  if (n >= 1_000) return `${n / 1_000}k`;
  return String(n);
}

export function SelectionFigure() {
  const [noise, setNoise] = useState(2);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let live = true;
    setError(null);
    getSelectionCurve(noise)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [noise]);

  if (error) {
    return html`<p class="note err">Could not compute the curve: ${error}</p>`;
  }
  if (!data) return html`<div class="empty">computing…</div>`;

  const rows = data.rows;
  const values = rows.flatMap((r) => SERIES.map((s) => r[s.key]));
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  // A perfect oracle collapses all three series onto one line, so the data range is zero and
  // every point would map to the same pixel (or divide by zero). Give it a window instead --
  // a flat line across the middle of the chart is the correct picture of "nothing happens".
  if (hi - lo < 0.5) {
    const mid = (hi + lo) / 2;
    lo = mid - 1;
    hi = mid + 1;
  } else {
    const margin = (hi - lo) * 0.08;
    lo -= margin;
    hi += margin;
  }

  // Equal spacing IS log spacing: the sizes are powers of ten.
  const x = (i) => PAD.left + (rows.length === 1 ? PLOT_W / 2 : (i * PLOT_W) / (rows.length - 1));
  // More negative is better, and better goes up. `lo` is the most negative value, so it maps
  // to the top of the plot -- the axis is inverted on purpose and the caption says so.
  const y = (v) => PAD.top + ((v - lo) / (hi - lo)) * PLOT_H;

  const path = (key) => rows.map((r, i) => `${i ? 'L' : 'M'}${x(i)},${y(r[key])}`).join(' ');

  // The band between what was reported and what was really picked: the part of the apparent
  // improvement that is the oracle being wrong.
  const band =
    rows.map((r, i) => `${i ? 'L' : 'M'}${x(i)},${y(r.best_measured)}`).join(' ') +
    ' ' +
    rows
      .slice()
      .reverse()
      .map((r, i) => `L${x(rows.length - 1 - i)},${y(r.true_of_pick)}`)
      .join(' ') +
    ' Z';

  const ticks = 5;
  const gridY = Array.from({ length: ticks }, (_, i) => lo + ((hi - lo) * i) / (ticks - 1));

  const first = rows[0];
  const last = rows[rows.length - 1];
  const reportedGain = first.best_measured - last.best_measured;
  const realGain = first.true_of_pick - last.true_of_pick;
  const perfect = data.assumptions.noise === 0;

  return html`
    <div class="selfig">
      <div class="selfig-controls">
        <span class="small muted">Oracle error (σ, kcal/mol)</span>
        ${NOISE_STEPS.map(
          (s) => html`
            <button
              type="button"
              key=${s.v}
              class=${'nstep' + (noise === s.v ? ' sel' : '')}
              title=${s.hint}
              onClick=${() => setNoise(s.v)}
            >
              ${s.label}
            </button>
          `,
        )}
      </div>

      <svg viewBox=${`0 0 ${W} ${H}`} class="selplot" role="img"
           aria-label="Reported score, true value of the selected molecule, and the pool's ceiling, against pool size">
        ${gridY.map(
          (v) => html`
            <g key=${v}>
              <line class="grid" x1=${PAD.left} x2=${W - PAD.right} y1=${y(v)} y2=${y(v)} />
              <text class="ylab" x=${PAD.left - 8} y=${y(v) + 3.5}>${v.toFixed(1)}</text>
            </g>
          `,
        )}

        ${!perfect && html`<path class="band" d=${band} />`}

        ${SERIES.map(
          (s) => html`<path key=${s.key} class=${'line ' + s.cls} d=${path(s.key)} />`,
        )}
        ${SERIES.map((s) =>
          rows.map(
            (r, i) => html`<circle key=${s.key + i} class=${'dot ' + s.cls}
                                   cx=${x(i)} cy=${y(r[s.key])} r="3" />`,
          ),
        )}

        ${rows.map(
          (r, i) => html`<text key=${r.n} class="xlab" x=${x(i)} y=${H - PAD.bottom + 18}
                               text-anchor="middle">${fmtN(r.n)}</text>`,
        )}
        <text class="axis" x=${PAD.left + PLOT_W / 2} y=${H - 6} text-anchor="middle">
          molecules screened
        </text>
        <text class="axis" transform=${`rotate(-90 12 ${PAD.top + PLOT_H / 2})`}
              x="12" y=${PAD.top + PLOT_H / 2} text-anchor="middle">
          kcal/mol (better is up)
        </text>
      </svg>

      <div class="selfig-legend">
        ${SERIES.map(
          (s) => html`<span class="lg" key=${s.key}>
            <i class=${'swatch ' + s.cls}></i><span class="lg-name">${s.name}</span>
            <em class="small muted">${s.note}</em>
          </span>`,
        )}
      </div>

      ${perfect
        ? html`<p class="note ok">
            With a perfect oracle the three series coincide. The best compound by measured
            score is always the best compound present, regret is zero at every pool size, and
            At σ = 0, increasing the pool size does not introduce selection error.
          </p>`
        : html`<p class="selfig-read">
            ${/* htm drops the whitespace between a text node and an element that follows it
                 on the next line, so every such boundary needs an explicit ${' '} or the
                 words run together. */ ''}
            Increasing the pool from ${fmtN(first.n)} to ${fmtN(last.n)} compounds improves
            the${' '}<b>reported</b> score by
            ${' '}<b>${reportedGain.toFixed(2)} kcal/mol</b> and the selected compound's true
            value by <b>${realGain.toFixed(2)}</b>. The shaded band is the difference.
            ${' '}Over the same range the gap to the pool's own ceiling <b>rises</b> from
            ${' '}${first.regret.toFixed(2)} to ${last.regret.toFixed(2)} kcal/mol: a larger
            pool contains more mediocre compounds able to outrank the best one by error alone.
          </p>`}

      <details class="selfig-detail">
        <summary>Values and assumptions</summary>
        <table class="cd-table">
          <thead>
            <tr>
              <th class="num">screened</th>
              <th class="num">reported</th>
              <th class="num">truly picked</th>
              <th class="num">pool ceiling</th>
              <th class="num">regret</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(
              (r) => html`<tr key=${r.n}>
                <td class="num mono">${fmtN(r.n)}</td>
                <td class="num mono">${r.best_measured.toFixed(2)}</td>
                <td class="num mono">${r.true_of_pick.toFixed(2)}</td>
                <td class="num mono">${r.best_true_available.toFixed(2)}</td>
                <td class="num mono">${r.regret.toFixed(2)}</td>
              </tr>`,
            )}
          </tbody>
        </table>
        <p class="small muted">
          Simulated. True affinity is drawn from a normal distribution centred on${' '}
          ${data.assumptions.center} kcal/mol with standard deviation
          ${' '}${data.assumptions.spread}, and scored by an oracle with independent error
          σ = ${data.assumptions.noise}. Values are averaged over
          ${' '}${data.assumptions.trials} pools per size at seed ${data.assumptions.seed},
          so the table reproduces exactly on a rerun. σ = 2 approximates the published
          affinity RMSE of AutoDock Vina, a literature figure rather than a measurement taken
          here. The model is the simplest one that exhibits the effect and does not represent
          any particular target.
        </p>
      </details>
    </div>
  `;
}
