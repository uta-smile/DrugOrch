/* The 3D pose viewer: what the docking score was actually computed from.
 *
 * A Vina score is a summary of a geometry, and the geometry is the thing that can be
 * wrong. A favourable number computed from a ligand wedged through a backbone is a false
 * positive that no filter on the number itself will catch, so the pose has to be
 * lookable-at. That is the whole argument for this panel.
 *
 * Vina returns an *ensemble* - nine poses per molecule here - ordered by score. Showing
 * only the top one hides useful signal: when the top poses disagree about where the ligand
 * sits, the score is less trustworthy than when they cluster. So the ensemble is steppable
 * and the count is stated rather than quietly collapsed to one.
 *
 * The poses arrive as SDF, converted server-side from the PDBQT Vina wrote. 3Dmol.js
 * accepts PDBQT without raising and returns a different molecule than the file holds - on
 * a 9-model, 27-atom file it reported 27 frames and 9 atoms of only C and N, dropping a
 * whole sulfonyl group - so the raw format must never reach this component.
 *
 * 3Dmol is imperative and owns its own canvas, so it sits behind a ref and outside preact's
 * render cycle. It also cannot read CSS variables - the reason `themechange` exists in
 * main.js - so colours are resolved from the computed style and reapplied on that event.
 */

import { useEffect, useRef, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { fetchPose, fetchReceptor, getPoses } from '@/api.js';

/* 3Dmol is ~540 KB. Loading it when the panel first opens rather than at module scope keeps
   it off the critical path for a run that never looks at a pose. The promise is cached, so
   remounting the panel does not refetch. */
let molPromise = null;
const loadMol = () => (molPromise ??= import('3dmol'));

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function PoseViewer({ runId }) {
  const [index, setIndex] = useState(null);
  const [selected, setSelected] = useState(0);
  const [frame, setFrame] = useState(0);
  const [nFrames, setNFrames] = useState(0);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState('');

  const hostRef = useRef(null);
  const viewerRef = useRef(null);
  const receptorRef = useRef(null);
  const ligandRef = useRef(null);
  // Guards against an out-of-order load: a fast second click must not be overwritten by a
  // slow first one finishing afterwards.
  const loadIdRef = useRef(0);

  useEffect(() => {
    let alive = true;
    setStatus('loading');
    getPoses(runId)
      .then((d) => {
        if (!alive) return;
        setIndex(d);
        setSelected(0);
        setStatus(d.n_with_file ? 'ready' : 'empty');
      })
      .catch((e) => {
        if (!alive) return;
        setError(String(e.message || e));
        setStatus('error');
      });
    return () => { alive = false; };
  }, [runId]);

  const withFile = (index?.poses || []).filter((p) => p.file);
  const current = withFile[selected] || null;

  /* The receptor is loaded once per run. It is ~2,400 atoms and its cartoon geometry is by
     far the most expensive thing here to build; rebuilding it on every pose selection froze
     the renderer outright, so it is deliberately kept out of the ligand effect below. */
  useEffect(() => {
    if (status !== 'ready' || !hostRef.current) return;
    let alive = true;

    (async () => {
      const $3Dmol = await loadMol();
      if (!alive || !hostRef.current) return;

      if (!viewerRef.current) {
        viewerRef.current = $3Dmol.createViewer(hostRef.current, {
          backgroundColor: cssVar('--bg-sunken', '#111'),
          antialias: true,
        });
      }
      const viewer = viewerRef.current;

      const receptor = await fetchReceptor(runId).catch(() => null);
      if (!alive || !receptor) return;

      receptorRef.current = viewer.addModel(receptor, 'pdb');
      viewer.setStyle(
        { model: receptorRef.current },
        { cartoon: { color: cssVar('--fg-faint', '#8a8f98') } },
      );
      viewer.render();
    })().catch(() => { /* no receptor is survivable: the ligand still renders alone */ });

    return () => { alive = false; };
  }, [status, runId]);

  /* The selected ligand's ensemble. Only this model is swapped. */
  useEffect(() => {
    if (status !== 'ready' || !current || !hostRef.current) return;
    const myLoad = ++loadIdRef.current;
    let alive = true;

    (async () => {
      await loadMol();
      const viewer = viewerRef.current;
      if (!alive || myLoad !== loadIdRef.current || !viewer) return;

      const pose = await fetchPose(runId, current.file);
      if (!alive || myLoad !== loadIdRef.current) return;

      if (ligandRef.current) viewer.removeModel(ligandRef.current);
      ligandRef.current = viewer.addModel(pose, 'sdf', { multimodel: true, frames: true });
      viewer.setStyle(
        { model: ligandRef.current },
        { stick: { radius: 0.18, colorscheme: 'greenCarbon' } },
      );
      viewer.zoomTo({ model: ligandRef.current });
      viewer.zoom(0.45);
      viewer.setFrame(0);
      viewer.render();
      setFrame(0);
      setNFrames(viewer.getNumFrames ? viewer.getNumFrames() : 0);
    })().catch((e) => {
      if (alive && myLoad === loadIdRef.current) setError(String(e.message || e));
    });

    return () => { alive = false; };
  }, [status, runId, current?.file]);

  /* Release the WebGL context on unmount. Browsers cap how many a page may hold, and a run
     view that is navigated in and out of would otherwise exhaust them. */
  useEffect(
    () => () => {
      const viewer = viewerRef.current;
      viewerRef.current = null;
      receptorRef.current = null;
      ligandRef.current = null;
      try { viewer?.clear?.(); } catch { /* already torn down */ }
    },
    [],
  );

  /* 3Dmol cannot read CSS variables, so the background is reapplied imperatively. This is
     the consumer main.js's themechange event was dispatched for. */
  useEffect(() => {
    function onTheme() {
      const viewer = viewerRef.current;
      if (!viewer) return;
      viewer.setBackgroundColor(cssVar('--bg-sunken', '#111'));
      viewer.render();
    }
    window.addEventListener('themechange', onTheme);
    return () => window.removeEventListener('themechange', onTheme);
  }, []);

  function step(delta) {
    const viewer = viewerRef.current;
    if (!viewer || nFrames < 2) return;
    const next = (frame + delta + nFrames) % nFrames;
    viewer.setFrame(next);
    viewer.render();
    setFrame(next);
  }

  const missing = (index?.poses || []).length - withFile.length;

  return html`
    <div class="card pose-card">
      <h3>
        3D pose
        <span class="spacer"></span>
        ${index?.receptor &&
        html`<span class="small muted" style="font-weight:400;text-transform:none">
          ${index.receptor.pdb_id} chain ${index.receptor.chain}
        </span>`}
      </h3>
      <div class="body flush">
        ${status === 'loading' && html`<div class="empty">Loading poses...</div>`}
        ${status === 'error' && html`<div class="empty">${error}</div>`}
        ${status === 'empty' &&
        html`<div class="empty">
          No poses yet. They appear once docking has run.
          ${missing > 0 &&
          html`<br /><span class="small"
            >${missing} molecule${missing === 1 ? ' was' : 's were'} scored before poses were
            kept, so ${missing === 1 ? 'it has' : 'they have'} no geometry to show.</span
          >`}
        </div>`}

        ${status === 'ready' &&
        html`
          <div class="pose-host" ref=${hostRef}></div>

          <div class="pose-controls small">
            <button
              class="btn ghost"
              onClick=${() => step(-1)}
              disabled=${nFrames < 2}
              title="Previous pose in the ensemble"
            >${'‹'}</button>
            <span class="muted">pose ${frame + 1}${nFrames ? ` / ${nFrames}` : ''}</span>
            <button
              class="btn ghost"
              onClick=${() => step(1)}
              disabled=${nFrames < 2}
              title="Next pose in the ensemble"
            >${'›'}</button>
            <span class="spacer"></span>
            ${current && html`<span class="score">${current.score.toFixed(2)}</span>`}
          </div>

          <div class="pose-list">
            ${withFile.map(
              (p, i) => html`
                <button
                  key=${p.file}
                  class=${'pose-item' + (i === selected ? ' on' : '')}
                  onClick=${() => setSelected(i)}
                  title=${p.smiles}
                >
                  <span class="score">${p.score.toFixed(2)}</span>
                  <span class="smi">${p.smiles}</span>
                </button>
              `,
            )}
          </div>
          ${missing > 0 &&
          html`<div class="small muted" style="padding:.4rem .6rem">
            ${missing} more scored molecule${missing === 1 ? '' : 's'} without a stored pose.
          </div>`}
        `}
      </div>
    </div>
  `;
}
