/* A metric name that explains itself.
 *
 * Two patterns were available and they suit different places, so both are here and the
 * choice is made by context rather than by preference.
 *
 * In a table, the question is "does this column mean what I think". A popover anchored to
 * the heading answers it with the number still on screen, which is the whole point: the
 * reader is comparing a definition against a value, and a modal would cover the value while
 * a jump to a glossary would throw away the scroll position in the table being read. So
 * every metric outside the article is a `MetricTerm`.
 *
 * Inside the goals article the document *is* the reference, so the footnote pattern applies:
 * the term links down to its glossary entry and the entry links back. That is linkable,
 * survives in-page search and printing, and needs no JavaScript at all.
 *
 * The definitions are fetched once per page from `/api/glossary` and shared by both. A
 * module-level promise rather than per-component state: twelve column headings in one table
 * would otherwise open twelve identical requests.
 */

import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { getGlossary } from '@/api.js';

let cached = null;

export function loadGlossary() {
  if (!cached) {
    cached = getGlossary().catch((e) => {
      // Do not memoise a failure: a glossary that failed once because the server was
      // restarting would stay broken for the life of the page.
      cached = null;
      throw e;
    });
  }
  return cached;
}

/* Resolve a rendered column heading to an entry. The aliases come from the server with the
 * definitions, so a heading spelled `best` or `LE` resolves without the frontend keeping its
 * own mapping to drift from the real one. */
export function resolveMetric(data, name) {
  if (!data || !name) return null;
  const key = String(name).trim().toLowerCase().replace(/[\s-]+/g, '_');
  const direct = data.metrics.find((m) => m.key === key);
  if (direct) return direct;
  const aliased = data.aliases[key];
  if (aliased) return data.metrics.find((m) => m.key === aliased) || null;
  return data.metrics.find((m) => m.term.toLowerCase() === String(name).trim().toLowerCase())
    || null;
}

const ARROW = { higher: '↑ higher is better', lower: '↓ lower is better', context: '' };

/** A metric name rendered as a button that opens its definition beside itself. */
export function MetricTerm({ name, label, children }) {
  const [data, setData] = useState(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState(null);
  const btn = useRef(null);
  const pop = useRef(null);

  useEffect(() => {
    let live = true;
    loadGlossary().then((d) => live && setData(d)).catch(() => {});
    return () => {
      live = false;
    };
  }, []);

  // Measured after paint, not before: the popover has to exist to know how wide it is, and
  // clamping to the viewport without its width pushes it off-screen on narrow windows.
  useLayoutEffect(() => {
    if (!open || !btn.current) return;
    const r = btn.current.getBoundingClientRect();
    const w = pop.current?.offsetWidth || 320;
    const h = pop.current?.offsetHeight || 220;
    const left = Math.max(8, Math.min(r.left, innerWidth - w - 8));
    // Prefer below; flip above when there is not room, which there usually is not for a
    // heading in the last rows of a long table.
    const below = r.bottom + 6;
    const top = below + h > innerHeight - 8 ? Math.max(8, r.top - h - 6) : below;
    setPos({ left, top });
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => {
      if (e.type === 'keydown' && e.key !== 'Escape') return;
      /* A scroll or a click *inside* the popover is the reader using it, not leaving it.
       * `.mpop-body` is capped at 50vh and scrolls, so without this guard a long entry
       * dismissed itself the moment anyone tried to read past the fold.
       *
       * The `instanceof Node` test is not defensive padding. A page scroll arrives with
       * `window` or `document` as its target, and `Node.contains(window)` throws a
       * TypeError, which aborted the handler before it could close anything: the first
       * version of this guard fixed the inner scroll and broke the outer one. */
      const t = e.target;
      if (t instanceof Node && (pop.current?.contains(t) || btn.current?.contains(t))) return;
      setOpen(false);
    };
    addEventListener('keydown', close);
    addEventListener('mousedown', close);
    // A popover anchored to an element must not stay behind when that element moves.
    addEventListener('scroll', close, true);
    addEventListener('resize', close);
    return () => {
      removeEventListener('keydown', close);
      removeEventListener('mousedown', close);
      removeEventListener('scroll', close, true);
      removeEventListener('resize', close);
    };
  }, [open]);

  const m = resolveMetric(data, name);
  const text = children ?? label ?? name;

  // An unknown metric renders as plain text rather than as a button that does nothing.
  if (!m) return html`<span>${text}</span>`;

  return html`
    <span class="mterm-wrap">
      <button
        type="button"
        class="mterm"
        ref=${btn}
        aria-expanded=${open}
        title="What is ${m.term}?"
        onClick=${(e) => {
          e.stopPropagation();
          setOpen(!open);
        }}
      >
        ${text}
      </button>
      ${open && html`
        <div class="mpop" ref=${pop} role="dialog" aria-label=${m.term}
             style=${pos ? `left:${pos.left}px;top:${pos.top}px` : 'opacity:0'}>
          <div class="mpop-head">
            <b>${m.term}</b>
            ${ARROW[m.direction] && html`<span class="mdir small">${ARROW[m.direction]}</span>`}
            <span class="spacer"></span>
            <button type="button" class="mpop-x" aria-label="Close"
                    onClick=${() => setOpen(false)}>×</button>
          </div>
          <div class="mpop-body">
            <p class="mshort">${m.short}</p>
            ${m.range && html`<dl class="mrow"><dt>Range</dt><dd>${m.range}</dd></dl>`}
            ${m.threshold && html`<dl class="mrow"><dt>Threshold</dt><dd>${m.threshold}</dd></dl>`}
            ${m.computed && html`<dl class="mrow"><dt>Computed</dt><dd>${m.computed}</dd></dl>`}
            <dl class="mrow"><dt>How to read it</dt><dd>${m.reading}</dd></dl>
            ${m.caution && html`<p class="mcaution"><b>Caution.</b> ${m.caution}</p>`}
            ${m.see_also?.length > 0 && html`
              <div class="msee small muted">
                See also:${' '}
                ${m.see_also.map((k, i) => {
                  const other = resolveMetric(data, k);
                  return html`<span key=${k}>${i ? ', ' : ''}<a href=${'#/goals#m-' + k}
                    onClick=${() => setOpen(false)}>${other ? other.term : k}</a></span>`;
                })}
              </div>`}
          </div>
          <a class="mpop-more" href=${'#/goals#m-' + m.key} onClick=${() => setOpen(false)}>
            Full entry in the metrics glossary
          </a>
        </div>`}
    </span>
  `;
}
