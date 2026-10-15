/* One run's derived state, folded from the event stream.
 *
 * Every panel reads from here, so the trace, the leaderboard and the budget meter can never
 * disagree about what step the run is on — they are all looking at the same fold of the same
 * events. The raw event list is kept too, because the trace layout is computed from it and
 * because replay has to produce an identical render.
 */

const MAX_FEED = 500; // rows kept in the DOM; the full list stays in memory

export function emptyRun(runId) {
  return {
    runId,
    status: 'pending',
    step: 0,
    goal: '',
    policy: '',
    seed: null,
    target: null,
    budget: { total: 0, spent: 0, remaining: 0, by_subsystem: {} },
    best: [],
    objective: 'vina_score',
    molecules: { total: 0 },
    facts: {},
    lineage: {},
    events: [],
    feed: [],
    nEvents: 0,
    finished: null,
    warnings: [],
    lastSeq: 0,
  };
}

/** Fold a batch of events into a new run object. Returns the same object if nothing applies. */
export function applyEvents(run, batch) {
  if (!batch.length) return run;
  const next = { ...run };
  next.events = run.events.concat(batch);
  next.nEvents = next.events.length;
  const feed = run.feed.slice();

  for (const ev of batch) {
    next.lastSeq = Math.max(next.lastSeq, ev.seq || 0);
    if (ev.budget) next.budget = ev.budget;
    if (typeof ev.step === 'number') next.step = Math.max(next.step, ev.step);
    if (ev.status) next.status = ev.status;

    const d = ev.data || {};
    switch (ev.type) {
      case 'run_started':
        next.goal = d.goal || next.goal;
        next.target = d.target || next.target;
        next.seed = d.seed ?? next.seed;
        next.policy = d.policy?.name || next.policy;
        next.objective = d.objective?.key || next.objective;
        next.capabilities = Object.fromEntries(
          (d.subsystems || []).map((c) => [c.key, c]),
        );
        next.status = 'running';
        break;
      case 'best_updated':
        next.best = d.top || next.best;
        break;
      case 'molecules_added':
        next.molecules = { ...next.molecules, total: d.n_total ?? next.molecules.total };
        // Keep the lineage the event now carries, keyed by SMILES. The leaderboard shows a
        // molecule's score; this is what lets it also show where the molecule came from.
        if (d.records?.length) {
          next.lineage = { ...next.lineage };
          for (const r of d.records) if (r.smiles) next.lineage[r.smiles] = r;
        }
        break;

      // Facts used to arrive with no event at all, so the target panel rendered a snapshot
      // taken at page load and never updated for the rest of the run.
      case 'facts_updated':
        next.facts = { ...next.facts, ...(d.facts || {}) };
        break;
      case 'call_finished':
        if (d.summary?.top?.length && !next.best.length) {
          next.best = d.summary.top.map((t) => ({
            smiles: t.smiles,
            value: t[Object.keys(t).find((k) => k !== 'smiles')],
          }));
        }
        break;
      case 'run_finished':
      case 'run_cancelled':
        next.finished = { code: d.code, reason: d.reason, elapsed: d.elapsed_s };
        next.best = d.best || next.best;
        next.status = ev.type === 'run_cancelled' ? 'cancelled' : (ev.status || 'done');
        break;
      case 'warning':
      case 'subsystem_error':
      case 'budget_block':
      case 'policy_fallback':
        next.warnings = next.warnings.concat([
          { step: ev.step, type: ev.type, text: d.reason || d.message || '' },
        ]);
        break;
      default:
        break;
    }

    const line = feedLine(ev);
    if (line) feed.push(line);
  }

  // Fixed-height rows plus a cap keeps a long run from growing the DOM without bound; the
  // complete history is still in next.events.
  next.feed = feed.length > MAX_FEED ? feed.slice(feed.length - MAX_FEED) : feed;
  next.feedTruncated = feed.length - next.feed.length;
  return next;
}

function feedLine(ev) {
  const d = ev.data || {};
  const t = (ev.ts || '').slice(11, 19);
  switch (ev.type) {
    case 'run_started':
      return { t, type: 'start', msg: `run started · ${d.goal || ''}` };
    case 'decision':
      return {
        t,
        type: 'decision',
        sub: d.subsystem,
        msg: d.action === 'stop' ? 'stop: ' + (d.rationale || '') : `route → ${d.subsystem}`,
      };
    case 'call_started':
      return { t, type: 'start', sub: d.subsystem, msg: `${d.subsystem} started` };
    case 'call_progress':
      return { t, type: 'progress', sub: d.subsystem, msg: d.message || '' };
    case 'call_finished':
      return {
        t,
        type: d.ok ? 'ok' : 'error',
        sub: d.subsystem,
        msg: `${d.subsystem} ${d.ok ? 'ok' : 'failed'} · ${(d.wall_s ?? 0).toFixed(1)}s · ${d.note || d.error || ''}`,
      };
    case 'molecules_added':
      return { t, type: 'mol', sub: d.subsystem, msg: `+${d.n_new} molecules (${d.n_total} total)` };
    case 'best_updated':
      return { t, type: 'best', msg: `new best ${d.objective} = ${d.best}` };
    case 'budget_block':
      return { t, type: 'warn', sub: d.subsystem, msg: `budget block: needs ${d.needed}, ${d.remaining} left` };
    case 'policy_fallback':
      return { t, type: 'warn', msg: `policy fallback: ${d.rationale || ''}` };
    case 'subsystem_error':
      return { t, type: 'error', sub: d.subsystem, msg: `${d.stage}: ${d.reason}` };
    case 'warning':
      return { t, type: 'warn', msg: d.message || '' };
    case 'run_finished':
      return { t, type: 'done', msg: `finished: ${d.reason || ''}` };
    case 'run_cancelled':
      return { t, type: 'warn', msg: 'cancelled' };
    default:
      return null;
  }
}
