/* Run setup: four numbered steps, in the order a run actually needs them.
 *
 * The earlier version explained itself at length — a paragraph on why the policy choice is
 * yours, another on provider trade-offs, a hint under every field. All of it true, none of
 * it readable: a wall of prose is skipped, so the warnings that mattered were skipped too.
 *
 * So the page states, and details wait behind a click. "You pick this" is a line, not a
 * paragraph. A policy's reasoning lives in its card. A provider's caveat lives in its panel.
 * What stays on screen is the state you need to see at a glance — ready or not, chosen or
 * not — carried by colour and position rather than by sentences.
 */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { createRun, getPolicies } from '@/api.js';
import { ProviderPanel } from '@/components/Provider.js';
import { PolicyCard } from '@/components/PolicyCard.js';
import { TargetPicker } from '@/components/TargetPicker.js';
import { ToolPicker } from '@/components/ToolPicker.js';
import { ToolAuthoring } from '@/components/ToolAuthoring.js';
import { PromptPanel } from '@/components/PromptPanel.js';
import { GOAL_HANDOFF } from '@/views/goals.js';

/* A goal picked on the Goals page is parked in sessionStorage and collected here. Read once
 * and cleared: it is a handoff, not a saved preference, and a goal that reappeared every
 * time you opened the page would quietly overwrite one you had typed. */
function initialGoal() {
  try {
    const parked = sessionStorage.getItem(GOAL_HANDOFF);
    if (parked) {
      sessionStorage.removeItem(GOAL_HANDOFF);
      return parked;
    }
  } catch {
    /* private mode: no handoff, just the default */
  }
  return 'find selective JAK2 inhibitors with good oral bioavailability';
}

export function SetupView({ manifest, keyStatus, onKeyChange, onStarted }) {
  const [goal, setGoal] = useState(initialGoal);
  const [target, setTarget] = useState(null);
  const [ready, setReady] = useState(null);
  const [policy, setPolicy] = useState('scripted');
  const [budget, setBudget] = useState(manifest?.defaults?.budget_units ?? 120);
  const [seed, setSeed] = useState(42);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [cards, setCards] = useState(null);
  const [openCard, setOpenCard] = useState(null);
  const [llm, setLlm] = useState(manifest?.llm || null);
  const subsystems = manifest?.subsystems || [];
  const [tools, setTools] = useState(
    () => manifest?.defaults?.tools || subsystems.map((s) => s.key),
  );
  const [prompt, setPrompt] = useState({ extra: '', override: '' });
  // Gates the tool-authoring fetch: a closed <details> still renders its children.
  const [authoringOpen, setAuthoringOpen] = useState(false);

  useEffect(() => {
    getPolicies()
      .then((d) => setCards(Object.fromEntries(d.policies.map((c) => [c.name, c]))))
      .catch(() => setCards({}));
  }, []);

  const policies = manifest?.policies || [];
  const llmReady = llm ? llm.ready : keyStatus ? keyStatus.has_key : !!manifest?.has_api_key;
  const blocked = (p) => p === 'llm' && !llmReady;
  const targetBad = ready && ready.uniprot === '' && !ready.errors?.search;
  const canStart =
    goal.trim() && target && !busy && !blocked(policy) && !targetBad && tools.length > 0;

  async function start(e) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const { run_id } = await createRun({
        goal: goal.trim(),
        target: target?.uniprot || target?.key || '',
        policy,
        budget_units: Number(budget),
        seed: Number(seed),
        tools,
        // Sent only for the arm that has a prompt. Attaching it to a rule-based run would
        // put text in the record that provably did nothing.
        ...(policy === 'llm'
          ? { prompt_extra: prompt.extra, prompt_override: prompt.override }
          : {}),
      });
      onStarted(run_id);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return html`
    <div class="setup">
      <form onSubmit=${start}>
        ${Step({
          n: 1,
          title: 'Goal',
          aside: policy === 'llm' ? 'the model reads this' : `${policy} never reads this`,
          done: !!goal.trim(),
          children: html`
            <textarea
              value=${goal}
              rows="2"
              onInput=${(e) => setGoal(e.target.value)}
              placeholder="e.g. find potent, selective inhibitors with good oral bioavailability"
            ></textarea>
            <p class="note">
              A goal states what would count as success. It only affects a run under the
              <b>llm</b> policy, which is the policy that reads it.
              ${' '}
              ${/* Parks the current text before leaving. Navigating remounts this view, so a
                   goal someone had typed would otherwise be thrown away by the trip to a page
                   whose whole purpose is to help them write one. */ ''}
              <a
                href="#/goals"
                onClick=${() => {
                  try {
                    if (goal.trim()) sessionStorage.setItem(GOAL_HANDOFF, goal);
                  } catch {
                    /* private mode: the trip still works, the draft just does not survive it */
                  }
                }}
                >What kinds of goal drug discovery has →</a
              >
            </p>
          `,
        })}

        ${Step({
          n: 2,
          title: 'Target',
          done: !!target && !targetBad,
          children: html`<${TargetPicker} onChange=${setTarget} onReadiness=${setReady} />`,
        })}

        ${Step({
          n: 3,
          title: 'Tools',
          aside: `${tools.length} of ${subsystems.length} available to the policy`,
          done: tools.length > 0,
          children: html`
            <${ToolPicker} subsystems=${subsystems} enabled=${tools} onChange=${setTools} />
            <details class="authoring-wrap"
                     onToggle=${(e) => setAuthoringOpen(e.currentTarget.open)}>
              <summary>Add a tool of your own</summary>
              <${ToolAuthoring} active=${authoringOpen} />
            </details>
          `,
        })}

        ${Step({
          n: 4,
          title: 'Routing policy',
          aside: 'this choice is the experiment',
          done: true,
          children: html`
            <div class="pol-grid">
              ${policies.map(
                (p) => html`
                  <button
                    type="button"
                    key=${p.name}
                    class=${'pol' + (policy === p.name ? ' sel' : '') + (blocked(p.name) ? ' off' : '')}
                    disabled=${blocked(p.name)}
                    onClick=${() => setPolicy(p.name)}
                    title=${p.description}
                  >
                    <i class=${'dot pol-' + p.name}></i>
                    <span class="n">${p.name}</span>
                    ${cards?.[p.name] &&
                    html`<span
                      class="q"
                      role="button"
                      tabindex="0"
                      title="What does this policy do?"
                      onClick=${(e) => {
                        e.stopPropagation();
                        setOpenCard(openCard === p.name ? null : p.name);
                      }}
                      >?</span
                    >`}
                  </button>
                `,
              )}
            </div>
            ${blocked('llm') &&
            html`<p class="note warn">llm needs a credential. Set one in step 5.</p>`}
            ${openCard && cards?.[openCard] &&
            html`<${PolicyCard} card=${cards[openCard]} open=${true}
                                onToggle=${() => setOpenCard(null)} />`}
          `,
        })}

        ${Step({
          n: 5,
          title: 'Model',
          aside: llm?.ready ? llm.detail : 'needed only for the llm policy',
          done: llmReady,
          children: !keyStatus?.unavailable
            ? html`<${ProviderPanel} keyStatus=${keyStatus} onKeyChange=${onKeyChange}
                                     llm=${llm} onProviderChange=${setLlm} />`
            : null,
        })}

        ${/* Only for the arm that has one. A prompt panel above a scripted run would be
              showing text that will never be sent. */ ''}
        ${policy === 'llm' &&
        Step({
          n: 6,
          title: 'Prompt',
          aside: prompt.override
            ? 'replaced; this is a different arm'
            : prompt.extra
              ? 'you added instructions'
              : 'exactly what the model will be told',
          done: true,
          children: html`<${PromptPanel}
            goal=${goal}
            target=${target?.uniprot || target?.key || ''}
            budget=${budget}
            tools=${tools}
            extra=${prompt.extra}
            override=${prompt.override}
            onChange=${setPrompt}
          />`,
        })}

        <div class="launch">
          <div class="nums">
            <label>
              <span>Budget (units)</span>
              <input type="number" min="1" step="10" value=${budget}
                     onInput=${(e) => setBudget(e.target.value)} />
            </label>
            <label>
              <span>Seed</span>
              <input type="number" value=${seed} onInput=${(e) => setSeed(e.target.value)} />
            </label>
          </div>
          <span class="spacer"></span>
          <button class="btn primary lg" type="submit" disabled=${!canStart}>
            ${busy ? 'Starting…' : 'Start run'}
          </button>
        </div>

        ${error && html`<p class="note err">${error}</p>`}
        ${!target && !error && html`<p class="note">Pick a target to start.</p>`}
      </form>
    </div>
  `;
}

/* A numbered step. The number and its state ring are the navigation: a glance down the
 * left edge says how far along you are, without a progress bar or a word of explanation. */
function Step({ n, title, aside, done, children }) {
  if (children == null) return null;
  return html`
    <section class=${'step' + (done ? ' done' : '')}>
      <div class="rail"><span class="num">${done ? '✓' : n}</span></div>
      <div class="content">
        <h2>
          ${title}
          ${aside && html`<small>${aside}</small>`}
        </h2>
        ${children}
      </div>
    </section>
  `;
}
