/* The API-key panel: enter a key in the browser instead of editing .env and restarting.
 *
 * Two deliberate choices here.
 *
 * The key is verified before it is accepted. Without that, a typo would not surface until
 * the first routing decision, where the orchestrator correctly treats an API failure as a
 * `policy_fallback` and carries on under the scripted policy — so the run would finish
 * looking normal while measuring the wrong arm. A key is cheap to check and expensive to
 * get wrong, so it is checked at entry.
 *
 * The key is never echoed back. The server returns only a mask, and the input is cleared on
 * success. This page gets screen-recorded for demos; a key sitting in a text input is the
 * kind of thing that ends up in a conference video.
 */

import { useEffect, useRef, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { clearApiKey, getKeyStatus, setApiKey } from '@/api.js';

export function ApiKeyPanel({ status, onChange, compact = false }) {
  const [key, setKey] = useState('');
  const [persist, setPersist] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const [err, setErr] = useState(null);
  const [offerUnverified, setOfferUnverified] = useState(false);
  const [open, setOpen] = useState(!compact);
  const inputRef = useRef(null);

  useEffect(() => {
    if (open && !status?.has_key) inputRef.current?.focus();
  }, [open, status?.has_key]);

  async function save(e, { verify = true } = {}) {
    e?.preventDefault();
    const value = key.trim();
    if (!value) return;
    setBusy(true);
    setErr(null);
    setMsg(null);
    setOfferUnverified(false);
    try {
      const next = await setApiKey(value, { persist, verify });
      setKey('');
      setMsg(next.persisted_to ? `${next.message} · saved to .env` : next.message);
      onChange?.(next);
    } catch (e2) {
      setErr(e2.message);
      // Only when the API could not be reached at all. A 401 is the user's to fix, and
      // offering to save it anyway would just move the failure into the middle of a run.
      setOfferUnverified(e2.reachable === false);
    } finally {
      setBusy(false);
    }
  }

  async function forget(alsoEnv) {
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      const next = await clearApiKey(alsoEnv);
      setMsg(alsoEnv ? 'key cleared from this session and from .env' : 'key cleared for this session');
      onChange?.(next);
    } catch (e2) {
      setErr(e2.message);
    } finally {
      setBusy(false);
    }
  }

  const has = status?.has_key;
  const source = status?.source;

  return html`
    <div class="keypanel">
      <div class="row" style="gap:8px">
        <span class=${'dot ' + (has ? 'ok' : 'off')}></span>
        <b class="small">Anthropic API key</b>
        ${has &&
        html`<span class="mono small muted">${status.masked}</span>
             <span class="pill ${source === 'session' ? 'done' : ''}">${
               source === 'session' ? 'this session' : '.env'
             }</span>`}
        ${!has && html`<span class="small muted">not set; the llm policy is unavailable</span>`}
        <span class="spacer"></span>
        ${compact &&
        html`<button class="btn ghost small" type="button" onClick=${() => setOpen(!open)}>
          ${open ? 'close' : has ? 'change' : 'add key'}
        </button>`}
      </div>

      ${open &&
      html`
        <form onSubmit=${save} style="margin-top:9px">
          <div class="row" style="gap:7px;align-items:stretch">
            <input
              ref=${inputRef}
              type="password"
              class="mono"
              value=${key}
              autocomplete="off"
              spellcheck="false"
              placeholder=${has ? 'paste a different key…' : 'sk-ant-…'}
              onInput=${(e) => setKey(e.target.value)}
              style="flex:1;min-width:0"
            />
            <button class="btn primary" type="submit" disabled=${busy || !key.trim()}>
              ${busy ? 'checking…' : 'Save'}
            </button>
          </div>

          <label class="row small muted" style="margin-top:7px;gap:6px;cursor:pointer">
            <input
              type="checkbox"
              style="width:auto;margin:0"
              checked=${persist}
              onChange=${(e) => setPersist(e.target.checked)}
            />
            also write it to <code class="mono">.env</code> so it survives a server restart
          </label>

          <div class="hint" style="margin-top:6px">
            The key is checked against the API before it is accepted, kept in the server
            process, and never written into run records, which store only${' '}
            <code class="mono">api_key_source</code>. Get one at ${' '}
            <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noreferrer"
               >console.anthropic.com</a>.
          </div>

          ${has &&
          html`<div class="row small" style="margin-top:8px;gap:10px">
            <button class="btn ghost small" type="button" disabled=${busy}
                    onClick=${() => forget(false)}>
              Clear for this session
            </button>
            ${status.env_has_key &&
            html`<button class="btn ghost small" type="button" disabled=${busy}
                         onClick=${() => forget(true)}>
              Clear from .env too
            </button>`}
          </div>`}
        </form>
      `}

      ${msg && html`<div class="small" style="color:var(--ok);margin-top:7px">${msg}</div>`}
      ${err && html`<div class="small" style="color:var(--err);margin-top:7px">
        ${err}
        ${offerUnverified &&
        html`<div style="margin-top:6px">
          <button class="btn ghost small" type="button" disabled=${busy}
                  onClick=${() => save(null, { verify: false })}>
            Save without checking
          </button>
          <span class="muted" style="margin-left:7px">
            use this when offline or behind a proxy. The key is accepted as given
          </span>
        </div>`}
      </div>`}
    </div>
  `;
}

/** Load key status once and keep it fresh after edits. Shared by the header and setup. */
export function useKeyStatus() {
  const [status, setStatus] = useState(null);
  useEffect(() => {
    getKeyStatus()
      .then(setStatus)
      // A 403 means the server is not on loopback; the panel simply does not apply.
      .catch(() => setStatus({ has_key: false, source: null, unavailable: true }));
  }, []);
  return [status, setStatus];
}
