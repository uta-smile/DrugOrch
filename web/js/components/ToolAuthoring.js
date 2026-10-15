/* How to add a tool of your own, exported from the running server.
 *
 * The plugin mechanism always accepted third-party subsystems. Nothing in the interface
 * said so, named the factory the loader looks for, or showed which directory is scanned, so
 * in practice the only people who could add a tool were the ones who had read the source.
 *
 * This panel is that contract, exported rather than described:
 *
 * - the template is a complete working subsystem, served from the server and tested against
 *   the real plugin loader, so what is copied here is what was proved to load;
 * - the contract table is generated from the `Capability` and `ParamSpec` dataclasses, so a
 *   field added to either appears here without anybody updating a document;
 * - the load failures are the important part. A tool with a syntax error is discovered,
 *   fails to import, and is then absent from the manifest, the prompt and the picker, with
 *   everything looking normal. This is the only place that says why.
 */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { getToolAuthoring } from '@/api.js';

const TABS = [
  ['template', 'Template'],
  ['contract', 'Contract'],
  ['where', 'Where it goes'],
];

export function ToolAuthoring({ active = true }) {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('template');
  const [copied, setCopied] = useState(false);

  /* Only fetch once the panel is actually open. Preact renders the children of a closed
   * `<details>`, so this effect used to run on every visit to the setup page, and
   * `/api/tools/authoring` builds a whole registry: entry-point discovery, a plugin-path
   * scan, five subsystems constructed and the value cache opened, for a panel most people
   * never expand. */
  useEffect(() => {
    if (!active) return undefined;
    let live = true;
    getToolAuthoring()
      .then((d) => live && setInfo(d))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [active]);

  useEffect(() => {
    if (!copied) return undefined;
    const t = setTimeout(() => setCopied(false), 1400);
    return () => clearTimeout(t);
  }, [copied]);

  if (!active) return null;
  if (error) return html`<p class="note err">Could not load the tool contract: ${error}</p>`;
  if (!info) return html`<div class="empty">loading…</div>`;

  const noPaths = info.plugin_paths.length === 0;

  /* `navigator.clipboard` is undefined outside a secure context, which is exactly the
   * `--host 0.0.0.0` case this server anticipates: reached over http://<lan-ip> the optional
   * chain short-circuited, no state changed, and the button did nothing with no explanation.
   * Falling back to selecting the template tells the reader what to do instead. */
  function copy() {
    if (!navigator.clipboard) {
      const pre = document.querySelector('.authcode');
      if (pre) {
        const r = document.createRange();
        r.selectNodeContents(pre);
        const sel = getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
      }
      setCopied('manual');
      return;
    }
    navigator.clipboard.writeText(info.template).then(
      () => setCopied('yes'),
      () => setCopied('manual'),
    );
  }

  function download() {
    const blob = new Blob([info.template], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = info.template_filename;
    a.click();
    // Revoking synchronously can pull the URL before the download is committed, which on
    // some browsers makes the button silently do nothing. One turn of the loop is enough.
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  return html`
    <div class="authoring">
      ${info.load_failures?.length > 0 &&
      html`<div class="note err" style="margin:0 0 10px">
        <b>${info.load_failures.length} tool(s) failed to load.</b> A tool that cannot be
        imported is absent from the picker and from the routing policy's prompt, with nothing
        else reporting it.
        <ul class="authfail">
          ${info.load_failures.map(
            (f) => html`<li key=${f.name + f.source}>
              <b class="mono">${f.name}</b>
              <span class="small muted mono"> ${f.source}</span>
              <div class="mono tiny">${f.error}</div>
            </li>`,
          )}
        </ul>
      </div>`}

      <div class="ptabs">
        ${TABS.map(
          ([k, label]) => html`
            <button type="button" key=${k} class=${'ptab' + (tab === k ? ' sel' : '')}
                    onClick=${() => setTab(k)}>${label}</button>
          `,
        )}
        <span class="spacer"></span>
        ${tab === 'template' &&
        html`<button type="button" class="btn ghost sm" onClick=${copy}
                     title=${copied === 'manual' ? 'Clipboard access needs a secure context' : ''}>
          ${copied === 'yes' ? 'copied' : copied === 'manual' ? 'selected, press Ctrl+C' : 'Copy'}
        </button>
        <button type="button" class="btn ghost sm" onClick=${download}>Download .py</button>`}
      </div>

      ${tab === 'template' &&
      html`<div>
        <p class="note">
          A complete subsystem, not a sketch. It declares two parameters, returns records and
          reports a failure rather than raising. Save it as
          ${' '}<b class="mono">${info.template_filename}</b> in a plugin directory and
          restart the server.
        </p>
        <pre class="authcode">${info.template}</pre>
      </div>`}

      ${tab === 'contract' &&
      html`<div>
        <p class="note">
          Generated from the dataclasses the orchestrator uses, so this cannot fall behind
          them. A subsystem is any object with these members; the factory must be called
          ${' '}<b class="mono">${info.factory_name}</b>.
        </p>

        <h5 class="authh">Required members</h5>
        <table class="cd-table">
          <tbody>
            ${Object.entries(info.contract.methods).map(
              ([sig, doc]) => html`<tr key=${sig}>
                <td class="mono" style="white-space:nowrap">${sig}</td>
                <td>${doc}</td>
              </tr>`,
            )}
          </tbody>
        </table>

        <h5 class="authh">Capability fields</h5>
        <table class="cd-table">
          <tbody>
            ${info.contract.capability_fields.map(
              (f) => html`<tr key=${f.name}>
                <td class="mono">${f.name}</td><td class="mono small muted">${f.type}</td>
              </tr>`,
            )}
          </tbody>
        </table>

        <h5 class="authh">ParamSpec fields</h5>
        <table class="cd-table">
          <tbody>
            ${info.contract.param_spec_fields.map(
              (f) => html`<tr key=${f.name}>
                <td class="mono">${f.name}</td><td class="mono small muted">${f.type}</td>
              </tr>`,
            )}
          </tbody>
        </table>

        <h5 class="authh">Record shape</h5>
        <p class="small muted">
          Each entry of <b class="mono">records</b> is
          ${' '}<span class="mono">${JSON.stringify(info.contract.record_shape)}</span>. The
          orchestrator merges any subsystem's records without knowing which produced them.
        </p>
      </div>`}

      ${tab === 'where' &&
      html`<div>
        ${noPaths
          ? html`<p class="note warn">
              No plugin directory is configured, so a file saved anywhere will not be found.
              Set <b class="mono">plugin_paths</b> in the server settings, then restart.
            </p>`
          : html`
            <p class="note">Scanned at startup, in this order:</p>
            <table class="cd-table">
              <tbody>
                ${info.plugin_paths.map(
                  (p) => html`<tr key=${p.path}>
                    <td class="mono tiny">${p.path}</td>
                    <td style="white-space:nowrap">
                      <span class=${'chip ' + (p.exists ? 'ok' : 'err')}>
                        ${p.exists ? 'exists' : 'missing'}
                      </span>
                      ${p.exists && !p.writable &&
                      html`<span class="chip err">not writable</span>`}
                    </td>
                  </tr>`,
                )}
              </tbody>
            </table>`}

        <p class="note">
          A package that declares an entry point in the
          ${' '}<b class="mono">${info.entry_point_group}</b> group is also discovered. The
          built-in subsystems load through that same path, so there is no privileged tier.
        </p>
        <p class="note warn"><b>No sandbox.</b> ${info.sandbox}</p>
        <p class="note">${info.reload}</p>

        ${info.allow_list?.length > 0 &&
        html`<p class="note warn">
          An allow-list is set, so only these load:
          ${' '}<b class="mono">${info.allow_list.join(', ')}</b>.
        </p>`}

        ${info.loaded?.length > 0 &&
        html`<div>
          <h5 class="authh">Loaded now</h5>
          <table class="cd-table">
            <tbody>
              ${info.loaded.map(
                (l) => html`<tr key=${l.key}>
                  <td class="mono">${l.key}</td>
                  <td class="small muted mono tiny">${l.source}</td>
                </tr>`,
              )}
            </tbody>
          </table>
        </div>`}
      </div>`}
    </div>
  `;
}
