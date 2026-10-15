/* Target picker: find a protein without leaving the page for UniProt, RCSB and ChEMBL.
 *
 * Two tiers, because they cost differently. Typing hits UniProt only, so it keeps up with a
 * keystroke. Picking fans out to RCSB and ChEMBL to answer the question that actually
 * matters — will a run on this protein be able to do anything — before a budget is spent
 * rather than after.
 *
 * The verdict is chips, not prose: structure, actives, docking, each either green or not.
 */

import { useEffect, useRef, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { searchTargets, suggestedTargets, targetReadiness } from '@/api.js';

const KINDS = {
  accession: 'UniProt accession',
  pdb: 'PDB structure ID',
  chembl: 'ChEMBL target ID',
  text: '',
};

export function TargetPicker({ value, onChange, onReadiness }) {
  const [query, setQuery] = useState(value || '');
  const [hits, setHits] = useState([]);
  const [suggested, setSuggested] = useState([]);
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(-1);
  const [searching, setSearching] = useState(false);
  const [kind, setKind] = useState('text');
  const [picked, setPicked] = useState(null);
  const [ready, setReady] = useState(null);
  const [checking, setChecking] = useState(false);
  const abortRef = useRef(null);
  const readyRef = useRef(null);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  useEffect(() => {
    suggestedTargets()
      .then((d) => setSuggested(d.suggested || []))
      .catch(() => setSuggested([]));
  }, []);

  // Debounced search; the in-flight request is aborted so a fast typist cannot have an
  // earlier response land after a later one and overwrite it.
  useEffect(() => {
    const q = query.trim();
    if (!q || picked?.key === q || picked?.uniprot === q) {
      setHits([]);
      return undefined;
    }
    const timer = setTimeout(async () => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setSearching(true);
      try {
        const d = await searchTargets(q, ctrl.signal);
        setHits(d.results || []);
        setKind(d.kind || 'text');
        setCursor(-1);
        setOpen(true);
      } catch (e) {
        if (e.name !== 'AbortError') setHits([]);
      } finally {
        setSearching(false);
      }
    }, 220);
    return () => clearTimeout(timer);
  }, [query]);

  function pick(hit) {
    const q = hit.uniprot || hit.key;
    setPicked(hit);
    setQuery(q);
    setOpen(false);
    setCursor(-1);
    onChange?.(hit);
    check(q);
  }

  async function check(q) {
    readyRef.current?.abort();
    const ctrl = new AbortController();
    readyRef.current = ctrl;
    setChecking(true);
    setReady(null);
    onReadiness?.(null);
    try {
      const r = await targetReadiness(q, ctrl.signal);
      setReady(r);
      onReadiness?.(r);
    } catch (e) {
      if (e.name !== 'AbortError') {
        const failed = { verdict: { level: 'error', text: e.message }, errors: {} };
        setReady(failed);
        onReadiness?.(failed);
      }
    } finally {
      setChecking(false);
    }
  }

  const rows = query.trim() ? hits : [];
  const showSuggestions = open && !query.trim() && suggested.length > 0;

  function onKeyDown(e) {
    const list = showSuggestions ? suggested : rows;
    if (!open || !list.length) {
      if (e.key === 'ArrowDown') {
        setOpen(true);
        e.preventDefault();
      }
      return;
    }
    if (e.key === 'ArrowDown') {
      setCursor((c) => Math.min(list.length - 1, c + 1));
      e.preventDefault();
    } else if (e.key === 'ArrowUp') {
      setCursor((c) => Math.max(0, c - 1));
      e.preventDefault();
    } else if (e.key === 'Enter') {
      // Enter with nothing highlighted should submit the form, not swallow the keystroke.
      if (cursor < 0) return;
      e.preventDefault();
      const row = list[cursor];
      pick(showSuggestions ? { key: row.uniprot, uniprot: row.uniprot, gene: row.gene,
                               name: row.name, label: `${row.gene}: ${row.name}` } : row);
    } else if (e.key === 'Escape') {
      setOpen(false);
      setCursor(-1);
    }
  }

  // Keep the highlighted row in view when arrowing past the fold.
  useEffect(() => {
    if (cursor < 0 || !listRef.current) return;
    listRef.current.children[cursor]?.scrollIntoView({ block: 'nearest' });
  }, [cursor]);

  return html`
    <div class="field ac">
      <div class="tp-input">
        <input
          ref=${inputRef}
          value=${query}
          onInput=${(e) => {
            setQuery(e.target.value);
            setPicked(null);
            setReady(null);
            onChange?.(null);
            onReadiness?.(null);
          }}
          onFocus=${() => setOpen(true)}
          onBlur=${() => setTimeout(() => setOpen(false), 180)}
          onKeyDown=${onKeyDown}
          placeholder="gene name, protein name, UniProt accession, PDB ID, or ChEMBL ID"
          autocomplete="off"
          spellcheck="false"
          role="combobox"
          aria-expanded=${open ? 'true' : 'false'}
        />
        ${searching && html`<span class="tp-spin" aria-label="searching"></span>`}
      </div>

      ${KINDS[kind] &&
      html`<div class="hint">reading it as a <b>${KINDS[kind]}</b></div>`}

      ${showSuggestions &&
      html`<div class="ac-list" ref=${listRef}>
        <div class="ac-head">Common targets</div>
        ${suggested.map(
          (s, i) => html`
            <div
              class=${'ac-item' + (i === cursor ? ' cur' : '')}
              key=${s.uniprot}
              onMouseEnter=${() => setCursor(i)}
              onMouseDown=${() =>
                pick({ key: s.uniprot, uniprot: s.uniprot, gene: s.gene, name: s.name,
                       label: `${s.gene}: ${s.name}` })}
            >
              <div class="t"><b class="gene">${s.gene}</b> ${s.name}</div>
              <div class="s">${s.why}</div>
            </div>
          `,
        )}
      </div>`}

      ${open && rows.length > 0 &&
      html`<div class="ac-list" ref=${listRef}>
        ${rows.map(
          (hit, i) => html`
            <div
              class=${'ac-item' + (i === cursor ? ' cur' : '')}
              key=${hit.key + i}
              onMouseEnter=${() => setCursor(i)}
              onMouseDown=${() => pick(hit)}
            >
              <div class="t">
                ${hit.gene && html`<b class="gene">${hit.gene}</b>`}
                <span>${hit.name || hit.label}</span>
                ${hit.reviewed && html`<span class="tag ok" title="Reviewed UniProt entry">reviewed</span>`}
                ${hit.n_pdb > 0
                  ? html`<span class="tag" title=${`${hit.n_pdb} PDB entries cross-referenced`}>
                      ${hit.n_pdb} structures
                    </span>`
                  : html`<span class="tag dim" title="No PDB cross-references; docking is unlikely">
                      no structures
                    </span>`}
              </div>
              <div class="s">
                ${hit.sublabel}
                ${hit.matched_on && html`<span class="via"> · matched via ${hit.matched_on}</span>`}
              </div>
            </div>
          `,
        )}
      </div>`}

      ${open && query.trim() && !searching && rows.length === 0 &&
      html`<div class="ac-list"><div class="ac-empty">
        No match. Try a gene (JAK2), accession (O60674) or PDB ID (7LL4).
      </div></div>`}

      ${(checking || ready) && html`<${ReadinessCard} ready=${ready} checking=${checking} />`}
    </div>
  `;
}

/* The verdict. This is the part that earns the extra round trip: a user can see that a
 * target has a 1.3 Å structure with a bound inhibitor and seventeen thousand measured
 * actives — or that it has neither — before committing a budget to it. */
function ReadinessCard({ ready, checking }) {
  if (checking && !ready) {
    return html`<div class="rdy checking"><span class="tp-spin"></span>checking RCSB and ChEMBL…</div>`;
  }
  if (!ready) return null;
  const v = ready.verdict || {};
  const best = ready.best_structure;
  if (!ready.uniprot) {
    return html`<div class="rdy error"><i class="dot err"></i>${v.text}</div>`;
  }

  return html`
    <div class=${'rdy ' + (v.level || '')}>
      <span class="who">
        <b>${ready.gene || ready.uniprot}</b>
        <span class="muted">${ready.organism || ''}</span>
      </span>
      ${cap('structure', ready.can_dock,
            best ? `${best.pdb_id}${best.resolution ? ` ${best.resolution.toFixed(2)}Å` : ''}` : 'none')}
      ${cap('actives', ready.can_generate,
            ready.n_actives ? ready.n_actives.toLocaleString() : '0')}
      ${ready.structures?.length > 1 &&
      html`<span class="alt" title=${ready.structures.map((s2) => s2.pdb_id).join(', ')}>
        +${ready.structures.length - 1} more
      </span>`}
      ${v.level !== 'ok' && html`<span class="why">${v.text}</span>`}
    </div>
  `;
}

function cap(label, ok, detail) {
  return html`<span class=${'chip ' + (ok ? 'ok' : 'off')} title=${label}>
    ${ok ? '✓' : '✕'} ${detail}
  </span>`;
}
