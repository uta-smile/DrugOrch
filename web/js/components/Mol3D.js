/* A 3D molecule viewer that takes structure text, not a run.
 *
 * `PoseViewer` renders the poses belonging to a run and fetches them by run id. The
 * workbench has no run — a conformer someone asked for exists only in the reply to that
 * request — so the same rendering is needed over text that is already in hand. Rather than
 * teach PoseViewer a second data source, this takes SDF and optional receptor PDB as props
 * and knows nothing about where either came from.
 *
 * The hard-won details from PoseViewer carry over, because they are properties of 3Dmol
 * rather than of runs:
 *
 *   - **Never hand it PDBQT.** 3Dmol parses it without raising and returns a different
 *     molecule than the file holds — on a 9-model, 27-atom file it reported 27 frames and
 *     9 atoms of only C and N, dropping a whole sulfonyl group. Everything here is SDF or
 *     PDB, converted server-side.
 *   - **Load the receptor once.** Rebuilding a 2,400-atom cartoon on every molecule change
 *     froze the renderer outright, so the receptor is its own effect keyed on its own text.
 *   - **Release the WebGL context on unmount.** Browsers cap how many a page may hold.
 *   - **3Dmol cannot read CSS variables**, which is why `themechange` exists in main.js.
 */

import { useEffect, useRef, useState } from 'preact/hooks';
import { html } from '@/h.js';

let molPromise = null;
const loadMol = () => (molPromise ??= import('3dmol'));

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function Mol3D({ sdf, receptor, height = 320, style = 'stick', label }) {
  const hostRef = useRef(null);
  const viewerRef = useRef(null);
  const receptorRef = useRef(null);
  const ligandRef = useRef(null);
  const loadIdRef = useRef(0);

  const [frame, setFrame] = useState(0);
  const [nFrames, setNFrames] = useState(0);
  const [err, setErr] = useState('');

  /* The receptor, once per structure. See the note above on why this is not in the ligand
     effect: it is the expensive geometry and it almost never changes. */
  useEffect(() => {
    if (!receptor || !hostRef.current) return undefined;
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
      if (receptorRef.current) viewer.removeModel(receptorRef.current);
      receptorRef.current = viewer.addModel(receptor, 'pdb');
      viewer.setStyle(
        { model: receptorRef.current },
        { cartoon: { color: cssVar('--fg-faint', '#8a8f98') } },
      );
      viewer.render();
    })().catch(() => { /* no receptor is survivable: the ligand still renders alone */ });
    return () => { alive = false; };
  }, [receptor]);

  /* The molecule. Multi-record SDF becomes a steppable ensemble, which is what a docking
     result is — showing only the top pose hides whether the poses agree. */
  useEffect(() => {
    if (!sdf || !hostRef.current) return undefined;
    const myLoad = ++loadIdRef.current;
    let alive = true;

    (async () => {
      const $3Dmol = await loadMol();
      if (!alive || myLoad !== loadIdRef.current || !hostRef.current) return;
      if (!viewerRef.current) {
        viewerRef.current = $3Dmol.createViewer(hostRef.current, {
          backgroundColor: cssVar('--bg-sunken', '#111'),
          antialias: true,
        });
      }
      const viewer = viewerRef.current;

      if (ligandRef.current) viewer.removeModel(ligandRef.current);
      ligandRef.current = viewer.addModel(sdf, 'sdf', { multimodel: true, frames: true });
      viewer.setStyle(
        { model: ligandRef.current },
        style === 'ballstick'
          ? { stick: { radius: 0.14, colorscheme: 'greenCarbon' },
              sphere: { scale: 0.24, colorscheme: 'greenCarbon' } }
          : { stick: { radius: 0.18, colorscheme: 'greenCarbon' } },
      );
      viewer.zoomTo({ model: ligandRef.current });
      viewer.zoom(receptor ? 0.45 : 0.85);
      viewer.setFrame(0);
      viewer.render();
      setFrame(0);
      setNFrames(viewer.getNumFrames ? viewer.getNumFrames() : 0);
      setErr('');
    })().catch((e) => {
      if (alive && myLoad === loadIdRef.current) setErr(String(e.message || e));
    });

    return () => { alive = false; };
  }, [sdf, style, !!receptor]);

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

  useEffect(() => {
    function onTheme() {
      const viewer = viewerRef.current;
      if (!viewer) return;
      viewer.setBackgroundColor(cssVar('--bg-sunken', '#111'));
      viewer.render();
    }
    addEventListener('themechange', onTheme);
    return () => removeEventListener('themechange', onTheme);
  }, []);

  function step(delta) {
    const viewer = viewerRef.current;
    if (!viewer || nFrames < 2) return;
    const next = (frame + delta + nFrames) % nFrames;
    viewer.setFrame(next);
    viewer.render();
    setFrame(next);
  }

  return html`
    <div class="mol3d">
      <div class="pose-host" ref=${hostRef} style=${`height:${height}px`}></div>
      ${err && html`<div class="small" style="color:var(--err);padding:4px 8px">${err}</div>`}
      <div class="pose-controls small">
        ${nFrames > 1
          ? html`
              <button class="btn ghost" onClick=${() => step(-1)}
                      title="Previous conformation">${'‹'}</button>
              <span class="muted">${label || 'pose'} ${frame + 1} / ${nFrames}</span>
              <button class="btn ghost" onClick=${() => step(1)}
                      title="Next conformation">${'›'}</button>
            `
          : html`<span class="muted">${label || (nFrames ? '1 conformer' : '')}</span>`}
        <span class="spacer"></span>
        <span class="faint tiny">drag to rotate · scroll to zoom</span>
      </div>
    </div>
  `;
}

/* Offer the structure as a file. A conformer that can only be looked at is of no use to
   anyone who wants to run their own calculation on it. */
export function downloadSdf(sdf, name) {
  const blob = new Blob([sdf], { type: 'chemical/x-mdl-sdfile' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${(name || 'molecule').replace(/[^\w.-]+/g, '_').slice(0, 60)}.sdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
