/* The routing trace, rendered as one SVG.
 *
 * x is cumulative budget spent; a block's width is what it cost. The viewBox is in budget
 * units, so the drawing scales to any container without recomputing the layout, and the
 * whole thing serializes straight into a paper figure.
 *
 * Rationale text deliberately never appears inside the SVG. At projector distance it would
 * be illegible and it would wreck the figure; it lives in the inspector below instead.
 */

import { html } from '@/h.js';
import { GEOM, laneY, layout, traceHeight } from '@/trace/layout.js';

/* A fixed-width viewBox with budget units mapped onto it, rather than a viewBox whose width
 * *is* the budget. Two reasons: the trace then fills its container whatever the budget was,
 * and two runs with different budgets share one normalised x-domain, which is what makes
 * them comparable side by side. */
const VIEW_W = 1000;
const LABEL_W = 86;
const RIGHT_PAD = 14;
const PLOT_W = VIEW_W - LABEL_W - RIGHT_PAD;
const BLOCK_GAP = 0.07; // fraction of a block's width; without it, consecutive calls read as one bar

export function Swimlane({ events, capabilities, selectedStep, onSelect, showGhosts = true }) {
  const L = layout(events, capabilities);
  if (!L.lanes.length) {
    return html`<div class="empty">
      No subsystem calls yet. The trace will update as the run proceeds.
    </div>`;
  }

  const h = traceHeight(L.lanes.length);
  const laneIndex = Object.fromEntries(L.lanes.map((l, i) => [l, i]));
  const sx = PLOT_W / Math.max(L.maxX, 1);
  const X = (v) => LABEL_W + v * sx;
  // A call that cost nothing still has to be visible and clickable, so widths have a floor;
  // the gap is subtracted so eight consecutive docking rounds read as eight blocks.
  const W = (units) => {
    const raw = units * sx;
    return Math.max(5, raw - Math.max(3, raw * BLOCK_GAP));
  };

  const ticksEvery = niceStep(L.maxX);
  const gridlines = [];
  for (let v = 0; v <= L.maxX - GEOM.padX; v += ticksEvery) gridlines.push(v);

  return html`
    <div class="trace-wrap">
      <svg
        class="trace-svg"
        viewBox="0 0 ${VIEW_W} ${h}"
        preserveAspectRatio="xMinYMin meet"
        style="aspect-ratio:${VIEW_W} / ${h}"
        role="img"
        aria-label="Routing trace: which subsystem ran, in what order, and at what cost."
      >
        ${L.lanes.map(
          (lane, i) => html`
            <g key=${'lane' + lane}>
              <rect
                class="trace-lane-bg"
                x=${LABEL_W} y=${GEOM.policyH + i * GEOM.laneH + 5}
                width=${PLOT_W} height=${GEOM.laneH - 10} rx="4"
              />
              <text class="trace-lane-label" x="4" y=${laneY(i) + 4}>${lane.slice(0, 10)}</text>
            </g>
          `,
        )}

        ${gridlines.map(
          (v) => html`
            <g key=${'g' + v}>
              <line class="trace-grid-line" x1=${X(v)} y1=${GEOM.policyH - 8}
                    x2=${X(v)} y2=${h - GEOM.axisH} />
              <text class="trace-axis-text" x=${X(v)} y=${h - GEOM.axisH + 20}
                    text-anchor="middle">${v}</text>
            </g>
          `,
        )}
        <text class="trace-axis-text" x=${X(0)} y=${h - 6}>budget units spent →</text>

        ${L.edges.map((e, i) => {
          const a = L.blocks[e.from];
          const b = L.blocks[e.to];
          if (!a || !b || laneIndex[a.subsystem] == null || laneIndex[b.subsystem] == null) {
            return null;
          }
          const x1 = X(a.x + a.w);
          const y1 = laneY(laneIndex[a.subsystem]);
          const x2 = X(b.x);
          const y2 = laneY(laneIndex[b.subsystem]);
          const mid = (x1 + x2) / 2;
          return html`<path key=${'e' + i} class="trace-edge"
            d=${`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`} />`;
        })}

        ${showGhosts &&
        L.ghosts.map((g, i) => {
          const li = laneIndex[g.subsystem];
          if (li == null) return null;
          const y = laneY(li) - GEOM.ghostH / 2;
          return html`
            <g class="trace-ghost" key=${'gh' + i} onClick=${() => onSelect?.(g.step)}>
              <title>
${g.subsystem}: rejected at step ${g.step} (score ${g.score.toFixed(2)})
${g.reason}</title>
              <rect x=${X(g.x)} y=${y} width=${g.w} height=${GEOM.ghostH} rx="2.5" />
            </g>
          `;
        })}

        ${L.blocks.map((b, i) => {
          const li = laneIndex[b.subsystem];
          if (li == null) return null;
          const y = laneY(li) - GEOM.blockH / 2;
          const cls = [
            'trace-block',
            b.running ? 'running' : '',
            b.ok === false ? 'failed' : '',
            b.step === selectedStep ? 'sel' : '',
          ].join(' ');
          return html`
            <g class=${cls} key=${'b' + i} onClick=${() => onSelect?.(b.step)}>
              <title>
${b.subsystem}: step ${b.step}
${b.running ? 'running…' : `${b.ok ? 'ok' : 'failed'} · ${(b.cost ?? b.w).toFixed(1)} units · ${b.wallS.toFixed(1)}s`}
${b.note || b.error || ''}</title>
              <rect class="blk" x=${X(b.x)} y=${y} width=${W(b.w)} height=${GEOM.blockH}
                    rx="4" fill=${`var(--sub-${b.subsystem}, var(--sub-unknown))`} />
              ${b.running &&
              html`<rect class="blkfill" x=${X(b.x)} y=${y}
                      width=${Math.max(2, W(b.w) * (1 - (b.fraction || 0)))}
                      height=${GEOM.blockH} rx="4" fill="var(--bg-elev)" />`}
              ${b.ok === false &&
              html`<text x=${X(b.x) + 2.5} y=${y + GEOM.blockH - 8}>!</text>`}
            </g>
          `;
        })}

        ${L.ticks.map((t, i) => {
          const cls = [
            'trace-tick',
            t.step === selectedStep ? 'sel' : '',
            t.blocked ? 'blocked' : '',
          ].join(' ');
          const x = X(t.x);
          return html`
            <g class=${cls} key=${'t' + i} onClick=${() => onSelect?.(t.step)}>
              <title>step ${t.step}: ${t.action === 'stop' ? 'stop' : t.subsystem}</title>
              <line x1=${x} y1=${GEOM.policyH - 10} x2=${x} y2=${h - GEOM.axisH} />
              <polygon points=${`${x - 5},${GEOM.policyH - 22} ${x + 5},${GEOM.policyH - 22} ${x},${GEOM.policyH - 11}`} />
              <text x=${x} y=${GEOM.policyH - 26} text-anchor="middle">${t.step}</text>
            </g>
          `;
        })}
      </svg>

      <div class="trace-legend">
        ${L.lanes.map(
          (l) => html`<span class="item" key=${'lg' + l}>
            <i class="swatch" style=${`background:var(--sub-${l},var(--sub-unknown))`}></i>${l}
          </span>`,
        )}
        <span class="item"><i class="swatch ghost"></i>considered, not chosen</span>
        <span class="item" style="margin-left:auto">
          block width = budget spent · ${L.steps} decisions
        </span>
      </div>
    </div>
  `;
}

/** A round-ish gridline interval for the current axis span. */
function niceStep(span) {
  const target = span / 8;
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(target, 1))));
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (pow * m >= target) return pow * m;
  }
  return pow * 10;
}
