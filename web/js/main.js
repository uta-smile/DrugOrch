/* App entry: load the manifest, route between setup / run / history via the URL hash. */

import { render } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { getManifest, listRuns } from '@/api.js';
import { useKeyStatus } from '@/components/ApiKey.js';
import { SetupView } from '@/views/setup.js';
import { RunView } from '@/views/run.js';
import { BenchView } from '@/views/bench.js';
import { ToolsView } from '@/views/tools.js';
import { GenerateView } from '@/views/generate.js';
import { GoalsView } from '@/views/goals.js';

/* A route, and optionally an element inside it: `#/goals#m-qed` is the goals view scrolled
 * to the ligand-efficiency entry.
 *
 * The second `#` is the separator. Without it this router matched the whole hash against a
 * fixed list and fell through to setup for anything else, so every in-page anchor on the
 * goals article -- the contents rail, the section links, the glossary cross-references --
 * navigated to the setup page instead of scrolling. They are the kind of link that is only
 * ever exercised by clicking, which is why scrolling to those ids programmatically in
 * testing reported everything as working.
 *
 * Native anchor scrolling cannot do this: the browser would look for an element whose id is
 * `/goals#m-qed`. The view scrolls to `anchor` itself once it has rendered.
 */
function parseHash() {
  const raw = location.hash.slice(1);
  const cut = raw.indexOf('#');
  const h = (cut === -1 ? raw : raw.slice(0, cut)).replace(/^\/?/, '');
  const anchor = cut === -1 ? '' : raw.slice(cut + 1);

  if (h.startsWith('run/')) return { view: 'run', runId: decodeURIComponent(h.slice(4)), anchor };
  if (h === 'runs') return { view: 'runs', anchor };
  if (h === 'bench') return { view: 'bench', anchor };
  if (h === 'tools') return { view: 'tools', anchor };
  if (h === 'generate') return { view: 'generate', anchor };
  if (h === 'goals') return { view: 'goals', anchor };
  return { view: 'setup', anchor };
}

function App() {
  const [route, setRoute] = useState(parseHash);
  const [manifest, setManifest] = useState(null);
  const [keyStatus, setKeyStatus] = useKeyStatus();
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(
    () => document.documentElement.getAttribute('data-theme') || 'dark',
  );

  useEffect(() => {
    const onHash = () => setRoute(parseHash());
    addEventListener('hashchange', onHash);
    return () => removeEventListener('hashchange', onHash);
  }, []);

  useEffect(() => {
    getManifest().then(setManifest).catch((e) => setError(e.message));
  }, []);

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    setTheme(next);
    try {
      localStorage.setItem('drugorch-theme', next);
    } catch { /* private mode: the toggle still works for this session */ }
    // 3Dmol and uPlot cannot read CSS variables, so anything imperative re-reads on this.
    dispatchEvent(new CustomEvent('themechange', { detail: next }));
  }

  const go = (hash) => {
    location.hash = hash;
  };

  return html`
    <div class="app">
      <header class="topbar">
        <span class="brand">Drug<span>Orch</span></span>
        <span class="small muted">Drug discovery runs with a live routing trace</span>
        <span class="spacer"></span>
        ${manifest?.llm &&
        html`<span class="keychip" title=${`${manifest.llm.provider}: ${manifest.llm.detail}`}>
          <span class=${'dot ' + (manifest.llm.ready ? 'ok' : 'off')}></span>
          <span class="small muted">${manifest.llm.ready ? manifest.llm.provider : 'no llm'}</span>
        </span>`}
        <button class="btn ghost" onClick=${() => go('#/generate')}
                title="Build conformers, elaborate a scaffold, score molecules against a pocket">
          Workbench
        </button>
        <button class="btn ghost" onClick=${() => go('#/goals')}
                title="What kinds of goal drug discovery has, and which ones a run can pursue">
          Goals
        </button>
        <button class="btn ghost" onClick=${() => go('#/bench')}>Leaderboard</button>
        <button class="btn ghost" onClick=${() => go('#/tools')}
                title="What the public benchmark gives its models, against what DrugOrch answers with">
          Toolsets
        </button>
        <button class="btn ghost" onClick=${() => go('#/runs')}>Runs</button>
        <button class="btn ghost" onClick=${() => go('#/')}>New run</button>
        <button class="btn ghost" onClick=${toggleTheme} title="Toggle theme" aria-label="Toggle theme">
          ${theme === 'dark' ? '☀' : '☾'}
        </button>
      </header>

      <main class="main">
        ${error && html`<div class="card"><div class="body" style="color:var(--err)">
          Could not load the manifest: ${error}
        </div></div>`}
        ${!manifest && !error && html`<div class="empty">loading…</div>`}
        ${manifest && route.view === 'setup' &&
        html`<${SetupView} manifest=${manifest} keyStatus=${keyStatus}
                           onKeyChange=${setKeyStatus}
                           onStarted=${(id) => go('#/run/' + id)} />`}
        ${manifest && route.view === 'run' &&
        html`<${RunView} runId=${route.runId} manifest=${manifest} onBack=${() => go('#/')} />`}
        ${manifest && route.view === 'runs' && html`<${RunsView} onOpen=${(id) => go('#/run/' + id)} />`}
        ${manifest && route.view === 'bench' && html`<${BenchView} />`}
        ${manifest && route.view === 'tools' && html`<${ToolsView} />`}
        ${route.view === 'generate' && html`<${GenerateView} />`}
        ${manifest && route.view === 'goals' &&
        html`<${GoalsView} manifest=${manifest} anchor=${route.anchor}
                           onUseGoal=${() => go('#/')} />`}
      </main>
    </div>
  `;
}

function RunsView({ onOpen }) {
  const [data, setData] = useState(null);
  useEffect(() => {
    listRuns().then(setData).catch(() => setData({ live: [], finished: [] }));
  }, []);
  if (!data) return html`<div class="empty">loading…</div>`;

  // A live run and its persisted copy are the same run; show the live one.
  const liveIds = new Set(data.live.map((r) => r.run_id));
  const rows = data.live.concat(data.finished.filter((r) => !liveIds.has(r.run_id)));

  if (!rows.length) {
    return html`<div class="card"><div class="empty">
      No runs yet. Start one from <b>New run</b>.
    </div></div>`;
  }

  return html`
    <div class="card">
      <h3>Runs <span class="spacer"></span>
        <span class="small muted" style="font-weight:400;text-transform:none">
          ${rows.length} run${rows.length === 1 ? '' : 's'}
        </span>
      </h3>
      <div class="body flush">
        <table class="runs-table">
          <thead>
            <tr>
              <th>run</th><th>policy</th><th>target</th><th>status</th>
              <th class="num">steps</th><th class="num">spent</th>
              <th class="num">molecules</th><th class="num">best</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(
              (r) => html`
                <tr key=${r.run_id} onClick=${() => onOpen(r.run_id)}>
                  <td class="mono small">${r.run_id}</td>
                  <td><span class="pill">${r.policy || '—'}</span></td>
                  <td class="small">${r.target || ''}</td>
                  <td><span class=${'pill ' + r.status}>${r.status}</span></td>
                  <td class="num">${r.steps}</td>
                  <td class="num">${(r.budget?.spent ?? 0).toFixed(0)}/${(r.budget?.total ?? 0).toFixed(0)}</td>
                  <td class="num">${r.n_molecules}</td>
                  <td class="num"><b>${r.best == null ? '—' : r.best.toFixed(2)}</b></td>
                </tr>
              `,
            )}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

render(html`<${App} />`, document.getElementById('app'));
