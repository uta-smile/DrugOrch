/* Which model routes, and how it is paid for.
 *
 * Three blurb-filled boxes became a segmented control. Each route's explanation is one
 * clause, and the one genuinely load-bearing warning — that switching provider changes the
 * arm under test — is a single line in a caution colour rather than a four-sentence
 * paragraph nobody finished reading.
 */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import {
  getChatgpt,
  getModels,
  setModelEffort,
  setProvider,
  signInChatgpt,
  signOutChatgpt,
} from '@/api.js';
import { ApiKeyPanel } from '@/components/ApiKey.js';

const PROVIDERS = [
  { id: 'chatgpt', title: 'ChatGPT plan', tag: 'no API key' },
  { id: 'anthropic', title: 'Claude', tag: 'API key' },
  { id: 'openai', title: 'OpenAI', tag: 'API key' },
];

export function ProviderPanel({ keyStatus, onKeyChange, llm, onProviderChange }) {
  const [provider, setLocal] = useState(llm?.provider || 'anthropic');
  const [chat, setChat] = useState(null);
  const [cat, setCat] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [note, setNote] = useState(null);

  useEffect(() => {
    getChatgpt().then(setChat).catch(() => setChat({ signed_in: false }));
  }, []);
  useEffect(() => {
    if (llm?.provider) setLocal(llm.provider);
  }, [llm?.provider]);

  useEffect(() => {
    let alive = true;
    setCat(null);
    getModels(provider)
      .then((d) => alive && setCat(d))
      .catch(() => alive && setCat({ models: [], efforts: [], source: 'unavailable' }));
    return () => {
      alive = false;
    };
  }, [provider, chat?.signed_in]);

  async function choose(id) {
    if (id === provider) return;
    setErr(null);
    setNote(null);
    setLocal(id);
    try {
      onProviderChange?.(await setProvider(id));
    } catch (e) {
      setErr(e.message);
    }
  }

  async function change(patch) {
    setErr(null);
    try {
      const next = await setModelEffort(patch);
      onProviderChange?.(next);
      setCat((c) => (c ? { ...c, current: { model: next.model, effort: next.effort } } : c));
      setNote(next.note || null);
    } catch (e) {
      setErr(e.message);
    }
  }

  async function signIn() {
    setBusy(true);
    setErr(null);
    setNote('Approve DrugOrch in the browser tab that just opened.');
    try {
      const next = await signInChatgpt();
      setChat(next);
      setNote(null);
      onProviderChange?.(await setProvider('chatgpt'));
      setLocal('chatgpt');
    } catch (e) {
      setErr(e.message);
      setNote(null);
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    setBusy(true);
    try {
      setChat(await signOutChatgpt());
      onProviderChange?.({ provider });
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  return html`
    <div class="prov">
      <div class="seg">
        ${PROVIDERS.map(
          (p) => html`
            <button type="button" key=${p.id}
                    class=${'seg-opt' + (provider === p.id ? ' sel' : '')}
                    onClick=${() => choose(p.id)}>
              ${p.title}<small>${p.tag}</small>
            </button>
          `,
        )}
      </div>

      ${provider === 'chatgpt' &&
      html`<div class="prov-auth">
        ${chat?.signed_in
          ? html`<i class="dot ok"></i>
              <b class="small">${chat.account_email || 'Signed in'}</b>
              ${!chat.authorized_for_plan &&
              html`<span class="chip err">plan not authorized</span>`}
              <span class="spacer"></span>
              <button class="btn ghost sm" type="button" disabled=${busy} onClick=${signOut}>
                Sign out
              </button>`
          : html`<i class="dot off"></i>
              <span class="small muted">Not signed in</span>
              <span class="spacer"></span>
              <button class="btn primary sm" type="button" disabled=${busy} onClick=${signIn}>
                ${busy ? 'Waiting…' : 'Sign in with ChatGPT'}
              </button>`}
      </div>`}

      ${provider === 'anthropic' &&
      html`<div class="prov-auth block">
        <${ApiKeyPanel} status=${keyStatus} onChange=${onKeyChange}
                        compact=${!!keyStatus?.has_key} />
      </div>`}

      ${provider === 'openai' &&
      html`<div class="prov-auth">
        <i class=${'dot ' + (llm?.ready ? 'ok' : 'off')}></i>
        <span class="small muted">
          ${llm?.ready ? 'Key configured' : html`Set <code class="mono">OPENAI_API_KEY</code> in .env`}
        </span>
      </div>`}

      ${ModelControls({ cat, llm, onChange: change })}

      ${provider !== 'anthropic' &&
      html`<p class="note warn">
        Different provider = different experiment. Recorded per run; don't pool results.
      </p>`}
      ${note && html`<p class="note ok">${note}</p>`}
      ${err && html`<p class="note err">${err}</p>`}
    </div>
  `;
}

/* Model and effort. Effort follows the model, because the vocabularies differ per model —
 * the dropdown shows what this one accepts rather than explaining that it varies. */
function ModelControls({ cat, llm, onChange }) {
  if (!cat) return html`<p class="note">loading models…</p>`;

  const model = cat.current?.model || llm?.model || '';
  const effort = cat.current?.effort || llm?.effort || cat.default_effort;
  const chosen = cat.models.find((m) => m.id === model);
  const efforts = chosen?.efforts?.length
    ? chosen.efforts.map((id) => ({ id, note: chosen.effort_notes?.[id] || '' }))
    : cat.efforts || [];

  return html`
    <div class="models">
      <label>
        <span>Model</span>
        <select value=${model} onChange=${(e) => onChange({ model: e.target.value })}>
          ${!chosen && model && html`<option value=${model}>${model} (unavailable)</option>`}
          ${cat.models.map((m) => html`<option key=${m.id} value=${m.id}>${m.label}</option>`)}
        </select>
      </label>
      <label>
        <span>Effort</span>
        <select value=${effort} onChange=${(e) => onChange({ effort: e.target.value })}>
          ${efforts.map((e) => html`<option key=${e.id} value=${e.id}>${e.id}</option>`)}
        </select>
      </label>
      ${cat.source === 'live' && html`<span class="chip ok">from your account</span>`}
      ${!chosen && model && html`<span class="chip err">pick an available model</span>`}
    </div>
  `;
}
