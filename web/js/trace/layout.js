/* Turn an event stream into trace geometry. Pure: events in, shapes out, no DOM.
 *
 * Keeping this separate from the renderer is what makes the trace testable -- the layout is
 * the part with actual logic in it, and it can be checked against a recorded run without a
 * browser.
 *
 * The x-axis is cumulative budget spent, not time. That is the whole idea: a block's width
 * is what it cost, so the trace *is* the budget, and two runs are comparable by shape. Using
 * wall-clock would make a cached docking round look free when it cost exactly as much as an
 * uncached one.
 *
 * Rejected alternatives are placed at their decision's x position as ghost blocks. They are
 * the evidence that routing happened at all: "it considered docking here and chose
 * generation instead" is a single visual fact, and no DAG or log view shows it.
 */

export const LANES = ['retrieval', 'properties', 'docking', 'generation'];

/* y-geometry is in viewBox units; x is mapped from budget units by the renderer, which is
 * the only part that knows how wide the drawing ends up. */
export const GEOM = {
  laneH: 82,        // height of one subsystem lane
  policyH: 50,      // the decision-tick strip above the lanes
  blockH: 46,       // a chosen invocation
  ghostH: 26,       // a rejected alternative
  padX: 2,          // budget units of breathing room at the right edge
  axisH: 44,
};

/** Cost a rejected alternative would have incurred, for its ghost width. */
function estimateCost(subsystem, capabilities) {
  const cap = capabilities?.[subsystem];
  if (!cap) return 1;
  // Per-molecule subsystems are quoted per call at a typical batch size, since the
  // alternative was never dispatched and we do not know how many molecules it would
  // have taken.
  return cap.per === 'molecule' ? cap.unit_cost * 8 : cap.unit_cost;
}

/**
 * @param {Array<object>} events  ordered event envelopes
 * @param {object} capabilities   {key: capability} from /api/manifest
 * @returns {{blocks,ghosts,ticks,edges,lanes,maxX,totalBudget,steps}}
 */
export function layout(events, capabilities = {}) {
  const blocks = [];
  const ghosts = [];
  const ticks = [];
  const edges = [];
  const byStep = new Map();

  let spent = 0;
  let totalBudget = 0;
  const seenLanes = new Set();
  const producedBy = new Map(); // subsystem -> last block index that produced molecules

  for (const ev of events) {
    if (ev.budget && typeof ev.budget.total === 'number') totalBudget = ev.budget.total;

    if (ev.type === 'decision') {
      const d = ev.data || {};
      const tick = {
        step: ev.step,
        x: spent,
        subsystem: d.subsystem || null,
        action: d.action,
        rationale: d.rationale || '',
        expectedGain: d.expected_gain || '',
        considered: d.considered || [],
        policy: d.policy || '',
        source: d.source || 'rule',
        latencyMs: d.latency_ms || 0,
        llm: d.llm || null,
      };
      ticks.push(tick);
      byStep.set(ev.step, tick);

      for (const alt of tick.considered) {
        if (!alt.subsystem || alt.subsystem === d.subsystem) continue;
        ghosts.push({
          step: ev.step,
          subsystem: alt.subsystem,
          x: spent,
          w: estimateCost(alt.subsystem, capabilities),
          score: alt.score ?? 0,
          reason: alt.reason || '',
        });
        seenLanes.add(alt.subsystem);
      }
    }

    if (ev.type === 'call_started') {
      const sub = ev.data?.subsystem;
      if (!sub) continue;
      seenLanes.add(sub);
      blocks.push({
        step: ev.step,
        subsystem: sub,
        x: spent,
        w: ev.data?.estimated_cost?.budget_units ?? 0,
        cost: 0,
        running: true,
        ok: null,
        fraction: 0,
        params: ev.data?.params || {},
        note: '',
        error: null,
        wallS: 0,
        nRecords: 0,
        cacheHits: 0,
      });
    }

    if (ev.type === 'call_progress') {
      const b = lastOpen(blocks, ev.data?.subsystem, ev.step);
      if (b) {
        b.fraction = ev.data?.fraction ?? b.fraction;
        b.message = ev.data?.message || '';
      }
    }

    if (ev.type === 'call_finished') {
      const d = ev.data || {};
      const b = lastOpen(blocks, d.subsystem, ev.step);
      const cost = d.cost?.budget_units ?? 0;
      if (b) {
        b.running = false;
        b.fraction = 1;
        b.ok = !!d.ok;
        b.cost = cost;
        b.w = cost;
        b.note = d.note || '';
        b.error = d.error || null;
        b.wallS = d.wall_s || 0;
        b.nRecords = d.summary?.n_records ?? 0;
        // How the call reached its result. The inspector renders this per
        // subsystem; without it a block says only how much it cost.
        b.explain = d.explain || null;
        b.cacheHits = d.cost?.cache_hits ?? 0;
        b.top = d.summary?.top || [];

        // A consumer depends on whatever last produced molecules. This is an approximation
        // of provenance -- the exact parent of each molecule is in molecules.csv -- but at
        // trace resolution "generation fed this docking round" is the useful statement.
        const src = producedBy.get('__last_producer__');
        if (src != null && src.subsystem !== d.subsystem && b.nRecords >= 0) {
          edges.push({ from: src.index, to: blocks.indexOf(b) });
        }
        if (b.nRecords > 0) {
          producedBy.set('__last_producer__', {
            index: blocks.indexOf(b),
            subsystem: d.subsystem,
          });
        }
      }
      spent += cost;
    }

    if (ev.type === 'budget_block') {
      const t = byStep.get(ev.step);
      if (t) {
        t.blocked = true;
        t.blockedReason =
          `needed ${ev.data?.needed ?? '?'} units, ${ev.data?.remaining ?? '?'} remained`;
      }
    }
  }

  const lanes = LANES.filter((l) => seenLanes.has(l)).concat(
    [...seenLanes].filter((l) => !LANES.includes(l)).sort(),
  );

  const maxX = Math.max(spent, totalBudget || 0, 1) + GEOM.padX;
  return { blocks, ghosts, ticks, edges, lanes, maxX, totalBudget, spent, steps: ticks.length };
}

function lastOpen(blocks, subsystem, step) {
  for (let i = blocks.length - 1; i >= 0; i--) {
    if (blocks[i].subsystem === subsystem && blocks[i].step === step) return blocks[i];
  }
  return null;
}

/** Pixel height of the whole trace for a given lane count. */
export function traceHeight(nLanes) {
  return GEOM.policyH + nLanes * GEOM.laneH + GEOM.axisH;
}

/** y of a lane's centre line. */
export function laneY(index) {
  return GEOM.policyH + index * GEOM.laneH + GEOM.laneH / 2;
}
