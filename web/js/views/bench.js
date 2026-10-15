/* Public benchmark results and within-project routing comparisons. */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { MetricTerm } from '@/components/MetricTerm.js';
import { getBench } from '@/api.js';
import { ExternalBoard } from '@/views/external.js';

const BOARDS = [
  ['external', 'Public benchmark', 'MolDesignBench, against published results'],
  ['routing', 'Routing arms', "this project's own benchmark, paired by target and seed"],
];

/** The two boards, public first. */
export function BenchView() {
  const [board, setBoard] = useState('external');
  return html`
    <div class="bench">
      <div class="seg boards" role="tablist">
        ${BOARDS.map(
          ([key, label, hint]) => html`<button
            key=${key}
            role="tab"
            aria-selected=${board === key}
            class=${'seg-btn' + (board === key ? ' on' : '')}
            onClick=${() => setBoard(key)}
            title=${hint}
          >
            ${label}
          </button>`,
        )}
      </div>
      ${board === 'external' ? html`<${ExternalBoard} />` : html`<${RoutingBoard} />`}
    </div>
  `;
}

function RoutingBoard() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [metricTab, setMetricTab] = useState('discovery_auc');

  useEffect(() => {
    getBench().then(setData).catch((e) => setErr(e.message));
  }, []);

  if (err) return card(html`<div class="note err">${err}</div>`);
  if (!data) return html`<div class="empty">loading…</div>`;

  if (!data.available) {
    return card(html`
      <div class="empty" style="text-align:left;line-height:1.7">
        <b>No benchmark has been run yet.</b><br />
        <span class="muted">
          Run the grid from the command line. Each arm uses the same withheld scaffolds for a
          given target and seed.
        </span>
        <pre class="mono tiny scroller" style="margin-top:10px">drugorch bench --targets O60674,P24941 --policies fixed,scripted,llm --seeds 1,2</pre>
      </div>
    `);
  }

  const board = data.leaderboard || [];
  const targets = Object.keys(data.per_target || {});
  const best = board[0];

  return html`
    <div>
      <div class="bench-head">
        <div>
          <h2>Routing arms</h2>
          <p class="muted small">
            ${data.n_cells} cells · ranked by <b class="mono">${data.metric}</b> ·${' '}
            ${data.metric_doc}
          </p>
        </div>
        <span class="spacer"></span>
        ${best && html`<div class="bench-winner">
          <div class="k">ahead</div>
          <div class="v"><i class=${'dot pol-' + best.policy}></i> ${best.policy}</div>
        </div>`}
      </div>

      ${board.length > 0 &&
      html`<table class="board">
        <thead>
          <tr>
            ${/* Every heading is a MetricTerm: clicking one opens its definition beside
                  the column, with the numbers still on screen. The `title` tooltips these
                  replace could not say how a metric is computed, what range is normal, or
                  what it is commonly misread as, and two of these columns are routinely
                  misread. */ ''}
            <th>arm</th>
            <th class="num"><${MetricTerm} name="discovery_auc" label="discovery AUC" /></th>
            <th class="num"><${MetricTerm} name="rediscovery" label="rediscovered" /></th>
            <th class="num"><${MetricTerm} name="best_score" label="best score" /></th>
            <th class="num"><${MetricTerm} name="novelty" label="novelty" /></th>
            <th class="num"><${MetricTerm} name="mean_delta" label="vs fixed" /></th>
            <th class="num"><${MetricTerm} name="win_rate" label="win rate" /></th>
            <th class="num" title="Paired target and seed combinations behind each row">cells</th>
          </tr>
        </thead>
        <tbody>
          ${board.map(
            (a, i) => html`
              <tr key=${a.policy} class=${i === 0 ? 'lead' : ''}>
                <td>
                  <i class=${'dot pol-' + a.policy}></i>
                  <b>${a.policy}</b>
                  ${a.flags.map(
                    (f, j) => html`<span class="flag" key=${'f' + j} title=${f}>!</span>`,
                  )}
                </td>
                <td class="num mono"><b>${a.primary.toFixed(4)}</b>
                  <span class="sd">±${a.primary_sd.toFixed(3)}</span>
                </td>
                <td class="num mono">${(a.rediscovery * 100).toFixed(1)}%</td>
                <td class="num mono">${a.best_score == null ? '—' : a.best_score.toFixed(2)}</td>
                <td class="num mono">${a.novelty.toFixed(2)}</td>
                <td class=${'num mono ' + deltaClass(a)}>
                  ${a.policy === 'fixed' ? '—' : fmtDelta(a.mean_delta)}
                  ${a.policy !== 'fixed' &&
                  html`<span class="sd">[${a.delta_ci[0].toFixed(2)}, ${a.delta_ci[1].toFixed(2)}]</span>`}
                </td>
                <td class="num mono">
                  ${a.policy === 'fixed' ? '—' : `${(a.win_rate * 100).toFixed(0)}%`}
                  ${a.policy !== 'fixed' &&
                  html`<span class="sd">${a.wins}W ${a.losses}L</span>`}
                </td>
                <td class="num mono">${a.n_cells}</td>
              </tr>
            `,
          )}
        </tbody>
      </table>`}

      ${board.some((a) => a.flags.length > 0) &&
      html`<div class="flags">
        ${board.flatMap((a) =>
          a.flags.map((f, j) => html`<div key=${a.policy + j} class="note warn">
            <b>${a.policy}</b>: ${f}
          </div>`),
        )}
      </div>`}

      ${targets.length > 0 &&
      html`<div class="card" style="margin-top:18px">
        <h3>Per target
          <span class="spacer"></span>
        </h3>
        <div class="body flush">
          <table class="board">
            <thead>
              <tr>
                <th>target</th>
                ${board.map((a) => html`<th class="num" key=${a.policy}>${a.policy}</th>`)}
              </tr>
            </thead>
            <tbody>
              ${targets.map((t) => {
                const row = data.per_target[t];
                const top = Math.max(...Object.values(row));
                return html`<tr key=${t}>
                  <td class="mono">${t}</td>
                  ${board.map(
                    (a) => html`<td class=${'num mono' + (row[a.policy] === top ? ' top' : '')}
                                    key=${a.policy}>
                      ${row[a.policy] == null ? '—' : row[a.policy].toFixed(3)}
                    </td>`,
                  )}
                </tr>`;
              })}
            </tbody>
          </table>
        </div>
      </div>`}

      ${data.failures?.length > 0 &&
      html`<div class="note warn" style="margin-top:14px">
        ${data.failures.length} cell(s) failed and are excluded:
        ${data.failures.slice(0, 4).map((f, i) => html`<div key=${'x' + i} class="mono tiny">
          ${f.target || '?'} ${f.policy || ''}: ${String(f.error).slice(0, 110)}
        </div>`)}
      </div>`}
    </div>
  `;
}

function fmtDelta(d) {
  return (d >= 0 ? '+' : '') + d.toFixed(3);
}

function deltaClass(a) {
  if (a.policy === 'fixed') return '';
  // Only colour it when the interval excludes zero. A positive mean with an interval
  // straddling zero is not a result, and colouring it green would say it was.
  if (a.delta_ci[0] > 0) return 'good';
  if (a.delta_ci[1] < 0) return 'bad';
  return '';
}

function card(children) {
  return html`<div class="card"><h3>Leaderboard</h3><div class="body">${children}</div></div>`;
}
