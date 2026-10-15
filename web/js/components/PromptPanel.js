/* The prompt, before the run — readable, editable, and not stored anywhere.
 *
 * The LLM log already shows what was sent, but only after a run has made calls, which is too
 * late to change anything and absent entirely for a run that never started. The question
 * "what is the model actually told?" should be answerable from the setup page, by reading,
 * not by starting a run and opening a drawer.
 *
 * Everything here is rendered server-side by the same functions the policy calls, so the
 * three blocks are the literal request rather than a description of it. The manifest in
 * particular is generated from the registry the run will be given: untick a tool and it
 * disappears from this text, because it will not be in the registry the policy sees.
 *
 * The edit box is the point of the panel rather than a decoration. A prompt that can only be
 * changed by editing a Python constant and restarting is one nobody tunes; one that lives in
 * an env var is one that silently applies to every later run and cannot be told apart in the
 * record. So this is per-run, sent with the run, recorded on each exchange, and part of the
 * LLM cache key — a changed prompt misses the cache rather than replaying a decision made
 * under different instructions.
 */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { previewPrompt } from '@/api.js';

export function PromptPanel({ goal, target, budget, tools, extra, override, onChange }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState('system');
  const [mode, setMode] = useState(override ? 'replace' : 'append');
  const [copied, setCopied] = useState('');

  const toolKey = (tools || []).join(',');

  useEffect(() => {
    let alive = true;
    // Debounced: the goal is a textarea and this re-renders a few KB of prompt per request.
    const t = setTimeout(() => {
      previewPrompt({
        goal,
        target,
        budget_units: Number(budget) || 120,
        tools,
        prompt_extra: extra,
        prompt_override: override,
      })
        .then((d) => alive && (setData(d), setErr(null)))
        .catch((e) => alive && setErr(e.message));
    }, 250);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [goal, target, budget, toolKey, extra, override]);

  if (err) return html`<p class="note err">Could not render the prompt: ${err}</p>`;
  if (!data) return html`<p class="note">Rendering the prompt…</p>`;

  const TABS = [
    ['system', 'System', data.system, 'The controller instructions. Everything else is data.'],
    ['manifest', 'Capabilities', data.manifest,
     'Generated from the tools this run gets. Unticking one removes it from this text.'],
    ['state', 'Run state', data.state,
     'RunState.summary(), as it will look at step 1. Every policy arm is offered this same view.'],
    ['schema', 'Response schema', JSON.stringify(data.schema, null, 2),
     'The model cannot name a subsystem outside this enum or invent a parameter.'],
  ];
  const current = TABS.find((t) => t[0] === tab) || TABS[0];

  async function copy(text, which) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(which);
      setTimeout(() => setCopied(''), 1400);
    } catch {
      setCopied('no');
    }
  }

  return html`
    <div class="prompt">
      <p class="note">${data.note}</p>

      <div class="ptabs">
        ${TABS.map(
          ([k, label, text]) => html`
            <button type="button" key=${k} class=${'ptab' + (k === tab ? ' sel' : '')}
                    onClick=${() => setTab(k)}>
              ${label}
              <span class="tiny faint">${Math.round(text.length / 4)} tok</span>
            </button>
          `,
        )}
        <span class="spacer"></span>
        ${data.kind !== 'standard' &&
        html`<span class=${'pill ' + (data.kind === 'replaced' ? 'failed' : '')}>
          prompt ${data.kind}
        </span>`}
        <button type="button" class="btn ghost small"
                onClick=${() => copy(current[2], tab)}>
          ${copied === tab ? 'copied' : 'copy'}
        </button>
      </div>

      <p class="small muted" style="margin:0 0 6px">${current[3]}</p>
      <pre class="read prompt-body">${current[2]}</pre>

      <div class="pedit">
        <div class="row" style="gap:10px;margin-bottom:5px">
          <b class="small">Your instructions</b>
          <label class="small muted">
            <input type="radio" name="pmode" checked=${mode === 'append'}
                   onChange=${() => { setMode('append'); onChange({ extra, override: '' }); }} />
            add to
          </label>
          <label class="small muted">
            <input type="radio" name="pmode" checked=${mode === 'replace'}
                   onChange=${() => { setMode('replace'); onChange({ extra: '', override: override || data.system }); }} />
            replace
          </label>
          <span class="spacer"></span>
          ${(extra || override) &&
          html`<button type="button" class="btn ghost small"
                       onClick=${() => onChange({ extra: '', override: '' })}>reset</button>`}
        </div>

        ${mode === 'append'
          ? html`<textarea
              rows="4"
              value=${extra}
              placeholder="Appended to the system text, e.g. 'This target has no crystal structure, so do not plan on docking.'"
              onInput=${(ev) => onChange({ extra: ev.target.value, override: '' })}
            ></textarea>`
          : html`<textarea
              rows="10"
              class="mono"
              value=${override}
              onInput=${(ev) => onChange({ extra: '', override: ev.target.value })}
            ></textarea>`}

        <p class="small muted" style="margin:5px 0 0">
          ${mode === 'replace'
            ? 'Replacing the controller instructions makes this a different arm rather than a tuned one. It will not be comparable to a standard run.'
            : 'Appending keeps the budget and precondition reasoning, and adds the new text after it.'}
          ${' '}Nothing is written to a file or to the environment; it applies to this run only.
        </p>
      </div>
    </div>
  `;
}
