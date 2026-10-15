/* The run view. Three zones answering three questions.
 *
 *   left   — what are we working on   (target, data-bank facts)
 *   centre — what did the orchestrator do, and why   (trace + inspector)
 *   right  — what did we get   (best molecules, LLM accounting, event feed)
 *
 * The LLM log is the exception to the three zones, and deliberately. Its warning — how many
 * decisions fell back — belongs in the right column where it cannot be missed, but the
 * exchanges themselves are several thousand tokens each and were unreadable in a 300px
 * column near the bottom of the page. So the card carries the warning and opens a drawer,
 * and the run bar has a button for it, because needing to scroll to find out what the
 * planner was asked defeats the point of recording it.
 *
 * Only the centre grows during a run, so the eye has one place to watch. While the run is
 * live the inspector follows the newest decision, which makes the page read as a narration
 * rather than something you have to drive; clicking any block pins it and stops the
 * following, because once you are investigating you do not want the view moving under you.
 */

import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { cancelRun, depictUrl, getRun } from '@/api.js';
import { RunStream } from '@/sse.js';
import { applyEvents, emptyRun } from '@/store.js';
import { Swimlane } from '@/trace/Swimlane.js';
import { Inspector } from '@/trace/Inspector.js';
import { layout } from '@/trace/layout.js';
import { LlmLogDrawer, LlmLogSummary } from '@/components/LlmLog.js';
import { PoseViewer } from '@/components/PoseViewer.js';

export function RunView({ runId, manifest, onBack }) {
  const [run, setRun] = useState(() => emptyRun(runId));
  const [conn, setConn] = useState('connecting');
  const [pinned, setPinned] = useState(null);
  const [showGhosts, setShowGhosts] = useState(true);
  const [logOpen, setLogOpen] = useState(false);
  const streamRef = useRef(null);

  useEffect(() => {
    setRun(emptyRun(runId));
    setPinned(null);
    setLogOpen(false);
    const stream = new RunStream(
      runId,
      (batch) => setRun((prev) => applyEvents(prev, batch)),
      setConn,
    );
    streamRef.current = stream;
    stream.start(0);
    // Snapshot in parallel: it fills in target details the event stream reports only once.
    getRun(runId)
      .then((snap) => setRun((prev) => ({ ...prev, snapshot: snap, goal: prev.goal || snap.goal })))
      .catch(() => {});
    return () => stream.close();
  }, [runId]);

  const caps = run.capabilities || manifestCaps(manifest);
  const L = useMemo(() => layout(run.events, caps), [run.nEvents, caps]);

  const latestStep = L.ticks.length ? L.ticks[L.ticks.length - 1].step : null;
  const selectedStep = pinned ?? latestStep;
  const tick = L.ticks.find((t) => t.step === selectedStep) || null;
  const block = L.blocks.find((b) => b.step === selectedStep) || null;

  // Only offer the viewer once docking has actually run: an empty 3D panel on a run that
  // never docked is a dead element the eye has to learn to skip.
  const hasDocked = L.blocks.some((b) => b.subsystem === 'docking');

  const live = run.status === 'running' || run.status === 'pending';
  const pct = run.budget.total ? (run.budget.spent / run.budget.total) * 100 : 0;
  const target = run.target || run.snapshot?.target;

  // Arrow keys step through decisions; a trace you can scrub is far easier to talk through.
  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      // The drawer binds its own arrows; scrubbing the trace underneath it would move a
      // view the reader cannot see.
      if (logOpen) return;
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      const steps = L.ticks.map((t) => t.step);
      if (!steps.length) return;
      const i = steps.indexOf(selectedStep);
      const next = e.key === 'ArrowLeft' ? Math.max(0, i - 1) : Math.min(steps.length - 1, i + 1);
      setPinned(steps[next]);
      e.preventDefault();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [L.ticks.length, selectedStep, logOpen]);

  return html`
    <div>
      <div class="runbar">
        <button class="btn ghost" onClick=${onBack} title="Back to setup">←</button>
        <span class="goal">${run.goal || runId}</span>
        ${target && html`<span class="pill">${target.label || target.query}</span>`}
        <span class=${'pill ' + run.status}>${run.status}</span>
        ${run.policy && html`<span class="pill">${run.policy}</span>`}
        <span class="mono small muted nowrap">
          ${run.budget.spent.toFixed(1)} / ${run.budget.total.toFixed(0)} units
        </span>
        <div class="meter" title=${`${pct.toFixed(0)}% of budget spent`}>
          <i style=${`width:${Math.min(100, pct)}%`}></i>
        </div>
        <span class="mono small muted nowrap">step ${run.step}</span>
        <button class="btn ghost small" onClick=${() => setLogOpen(true)}
                title="What the planner was asked and what it answered">
          LLM log
        </button>
        ${live &&
        html`<button class="btn danger" onClick=${() => cancelRun(runId).catch(() => {})}>
          Stop
        </button>`}
      </div>

      ${conn === 'reconnecting' &&
      html`<div class="conn">Connection lost. Reconnecting; the trace will resume where it stopped.</div>`}

      ${run.finished &&
      html`<div class="card" style="margin-bottom:14px">
        <div class="body">
          <b>Run ${run.status}</b> after ${run.step} steps: ${run.finished.reason}
          ${run.budget.remaining > 0.05 &&
          !/unspent/i.test(run.finished.reason || '') &&
          html`<span class="muted">
            · ${run.budget.remaining.toFixed(1)} units left unspent</span>`}
        </div>
      </div>`}

      <div class="cols">
        <div class="stack">
          ${TargetPanel({ target, snapshot: run.snapshot, facts: run.snapshot?.facts })}
          ${BudgetPanel({ run })}
        </div>

        <div class="stack">
          <div class="card">
            <h3>
              Routing trace
              <span class="spacer"></span>
              <label class="small muted" style="font-weight:400;text-transform:none;letter-spacing:0">
                <input type="checkbox" style="width:auto;margin-right:5px"
                       checked=${showGhosts} onChange=${(e) => setShowGhosts(e.target.checked)} />
                show rejected
              </label>
            </h3>
            <div class="body flush">
              ${Swimlane({
                events: run.events,
                capabilities: caps,
                selectedStep,
                onSelect: (s) => setPinned(s === pinned ? null : s),
                showGhosts,
              })}
            </div>
          </div>

          <div class="card">
            <h3>
              Why this decision
              <span class="spacer"></span>
              ${pinned != null
                ? html`<button class="btn ghost small" onClick=${() => setPinned(null)}>
                    following latest
                  </button>`
                : html`<span class="small muted" style="font-weight:400;text-transform:none">
                    following latest · ← → to step
                  </span>`}
            </h3>
            <div class="body">
              ${Inspector({ tick, block, capabilities: caps, runId })}
            </div>
          </div>
        </div>

        <div class="stack">
          ${Leaderboard({ run })}
          ${hasDocked && html`<${PoseViewer} runId=${runId} key=${runId} />`}
          <${LlmLogSummary} runId=${runId} nEvents=${run.nEvents}
                            onOpen=${() => setLogOpen(true)} />
          ${Feed({ run })}
        </div>
      </div>

      ${logOpen &&
      html`<${LlmLogDrawer} runId=${runId} nEvents=${run.nEvents}
                            onClose=${() => setLogOpen(false)} />`}
    </div>
  `;
}

function manifestCaps(manifest) {
  return Object.fromEntries((manifest?.subsystems || []).map((c) => [c.key, c]));
}

function TargetPanel({ target, snapshot, facts }) {
  const ident = facts?.target_identity;
  const shortlist = facts?.pdb_shortlist || [];
  const receptor = facts?.receptor;
  return html`
    <div class="card">
      <h3>Target</h3>
      <div class="body">
        ${!target && html`<div class="muted small">resolving…</div>`}
        ${target &&
        html`
          <div style="font-weight:650;margin-bottom:2px">
            ${ident?.gene || target.gene || target.query}
          </div>
          <div class="small muted" style="margin-bottom:9px">${ident?.name || target.name || ''}</div>
          <dl class="kv">
            ${target.uniprot && html`<dt>UniProt</dt><dd>${target.uniprot}</dd>`}
            ${target.chembl_id && html`<dt>ChEMBL</dt><dd>${target.chembl_id}</dd>`}
            ${ident?.organism && html`<dt>organism</dt><dd>${ident.organism}</dd>`}
            ${ident?.length && html`<dt>length</dt><dd>${ident.length} aa</dd>`}
          </dl>
        `}
        ${shortlist.length > 0 &&
        html`<div style="margin-top:11px">
          <div class="lbl small muted" style="margin-bottom:4px">Dockable structures</div>
          ${shortlist.slice(0, 4).map(
            (c, i) => html`<div class="small" key=${'p' + i} style="margin-bottom:3px">
              <b class="mono">${c.pdb_id}</b>
              ${receptor?.pdb_id === c.pdb_id
                ? html`<span class="pill done" style="margin-left:5px">in use</span>`
                : ''}
              <div class="faint" style="font-size:11px">${c.reason}</div>
            </div>`,
          )}
        </div>`}
        ${receptor &&
        html`<div class="small muted" style="margin-top:9px">
          Docking into <b class="mono">${receptor.pdb_id}</b> chain ${receptor.chain},
          box ${receptor.box.size_x.toFixed(0)}×${receptor.box.size_y.toFixed(0)}×${receptor.box.size_z.toFixed(0)} Å
          around ligand ${receptor.ligand_code}.
        </div>`}
      </div>
    </div>
  `;
}

function BudgetPanel({ run }) {
  const by = run.budget.by_subsystem || {};
  const entries = Object.entries(by).sort((a, b) => b[1] - a[1]);
  const total = run.budget.spent || 1;
  return html`
    <div class="card">
      <h3>Where the budget went</h3>
      <div class="body">
        ${entries.length === 0 && html`<div class="muted small">No subsystem calls yet.</div>`}
        ${entries.map(
          ([k, v]) => html`
            <div key=${k} style="margin-bottom:7px">
              <div class="row small" style="justify-content:space-between">
                <span class="row"><i class=${'dot sub-' + k}></i> ${k}</span>
                <span class="mono">${v.toFixed(1)}u</span>
              </div>
              <div class="meter" style="margin-top:3px;height:5px">
                <i style=${`width:${(v / total) * 100}%;background:var(--sub-${k},var(--sub-unknown))`}></i>
              </div>
            </div>
          `,
        )}
      </div>
    </div>
  `;
}

function Leaderboard({ run }) {
  const best = run.best || [];
  return html`
    <div class="card">
      <h3>Best molecules <span class="spacer"></span>
        <span class="small muted" style="font-weight:400;text-transform:none">${run.objective}</span>
      </h3>
      <div class="body flush">
        ${best.length === 0 &&
        html`<div class="empty">No scored molecules yet. Results appear after a scoring call.</div>`}
        ${best.slice(0, 8).map(
          (m, i) => html`
            <div class="mol" key=${'m' + i}>
              <img src=${depictUrl(m.smiles, 192, 144)} alt="" loading="lazy" width="96" height="72" />
              <div class="meta">
                <div class="score">${fmt(m.value)}</div>
                <div class="smi">${m.smiles}</div>
              </div>
            </div>
          `,
        )}
      </div>
    </div>
  `;
}

function Feed({ run }) {
  const ref = useRef(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [run.feed.length]);
  return html`
    <div class="card">
      <h3>Event feed <span class="spacer"></span>
        <span class="small muted" style="font-weight:400;text-transform:none">${run.nEvents}</span>
      </h3>
      <div
        class="body flush feed"
        ref=${ref}
        onScroll=${(e) => {
          const el = e.target;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
      >
        ${run.feedTruncated > 0 &&
        html`<div class="more">… ${run.feedTruncated} earlier events</div>`}
        ${run.feed.map(
          (f, i) => html`
            <div class="ev" key=${'f' + i}>
              <span class="t">${f.t}</span>
              <span class="ty" style=${f.sub ? `color:var(--sub-${f.sub},var(--fg-muted))` : ''}>
                ${f.type}
              </span>
              <span class="m" style=${tint(f.type)}>${f.msg}</span>
            </div>
          `,
        )}
      </div>
    </div>
  `;
}

function tint(type) {
  if (type === 'error') return 'color:var(--err)';
  if (type === 'warn') return 'color:var(--warn)';
  if (type === 'best') return 'color:var(--ok)';
  return '';
}

function fmt(v) {
  return typeof v === 'number' ? v.toFixed(2) : String(v ?? '—');
}
