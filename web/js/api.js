/* Thin wrappers over the HTTP API. Every call returns parsed JSON or throws. */

async function jfetch(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch { /* not JSON; the status text will do */ }
    // A detail can be a string or an object. Objects carry fields the caller acts on --
    // the key panel needs to know whether a failure was the key or the network -- so they
    // are hung off the Error rather than flattened into its message.
    if (detail && typeof detail === 'object') {
      const err = new Error(detail.message || res.statusText);
      Object.assign(err, detail);
      throw err;
    }
    throw new Error(detail);
  }
  return res.json();
}

export const getManifest = () => jfetch('/api/manifest');

export const searchTargets = (q, signal) =>
  jfetch(`/api/targets/search?q=${encodeURIComponent(q)}`, { signal });

export const suggestedTargets = () => jfetch('/api/targets/suggested');

export const targetReadiness = (q, signal) =>
  jfetch(`/api/targets/readiness?q=${encodeURIComponent(q)}`, { signal });

export const createRun = (spec) =>
  jfetch('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(spec),
  });

export const getRun = (id) => jfetch(`/api/runs/${encodeURIComponent(id)}`);
export const listRuns = () => jfetch('/api/runs');
export const cancelRun = (id) =>
  jfetch(`/api/runs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });

export const depictUrl = (smiles, w = 200, h = 150) =>
  `/api/depict?smiles=${encodeURIComponent(smiles)}&w=${w}&h=${h}`;

/* 3D. The pose and receptor endpoints return structure text, not JSON, so they bypass
   jfetch -- a viewer wants the file body, and parsing it here would only mean serialising
   it again on the way into the renderer. */
export const getPoses = (id) => jfetch(`/api/runs/${encodeURIComponent(id)}/poses`);

/* Default to SDF. 3Dmol.js parses PDBQT without complaining and produces the wrong
   molecule -- it transposes models and atoms -- so the page must never be handed the raw
   file. `format=pdbqt` exists for download and provenance, not for rendering. */
export const poseUrl = (id, file, format = 'sdf') =>
  `/api/runs/${encodeURIComponent(id)}/poses/${encodeURIComponent(file)}?format=${format}`;

export const receptorUrl = (id) => `/api/runs/${encodeURIComponent(id)}/receptor`;

async function text(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.text();
}

export const fetchPose = (id, file) => text(poseUrl(id, file));
export const fetchReceptor = (id) => text(receptorUrl(id));

export const getKeyStatus = () => jfetch('/api/key');

export const setApiKey = (key, { persist = false, verify = true } = {}) =>
  jfetch('/api/key', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, persist, verify }),
  });

export const clearApiKey = (forget = false) =>
  jfetch(`/api/key?forget=${forget ? 'true' : 'false'}`, { method: 'DELETE' });

export const getPolicies = () => jfetch('/api/policies');

export const getDecisions = (id, step) =>
  jfetch(`/api/runs/${encodeURIComponent(id)}/decisions${step ? `?step=${step}` : ''}`);

export const getLlmLog = (id, full = false) =>
  jfetch(`/api/runs/${encodeURIComponent(id)}/llm?full=${full ? 'true' : 'false'}`);

export const getChatgpt = () => jfetch('/api/chatgpt');
export const signInChatgpt = () => jfetch('/api/chatgpt/signin', { method: 'POST' });
export const signOutChatgpt = () => jfetch('/api/chatgpt', { method: 'DELETE' });
export const setProvider = (provider) =>
  jfetch('/api/provider', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider }),
  });

export const getModels = (provider) =>
  jfetch(`/api/models${provider ? `?provider=${encodeURIComponent(provider)}` : ''}`);

export const setModelEffort = (body) =>
  jfetch('/api/provider', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

export const getBench = () => jfetch('/api/bench');
export const getExternalBench = () => jfetch('/api/bench/external');
export const getBenchTools = () => jfetch('/api/bench/tools');

/* The generation workbench. These make no run: a run is a budgeted, routed experiment and
   these are single operations on molecules. */
const post = (url, body) =>
  jfetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

export const genEmbed = (smiles, seed = 42) => post('/api/gen/embed', { smiles, seed });
export const genElaborate = (spec) => post('/api/gen/elaborate', spec);
export const genPocket = (pdb_id, ligand_code = '') =>
  post('/api/gen/pocket', { pdb_id, ligand_code });
export const genDock = (spec) => post('/api/gen/dock', spec);
export const genJob = (id) => jfetch(`/api/gen/jobs/${encodeURIComponent(id)}`);
export const genJobCancel = (id) =>
  jfetch(`/api/gen/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' });

/* The prompt preview. A POST because the body carries the operator's draft text -- nothing
   is stored server-side, so there is no GET that could return it. */
export const previewPrompt = (spec) =>
  jfetch('/api/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(spec),
  });

export const getSelectionCurve = (noise, spread = 1.0) =>
  jfetch(`/api/selection-curve?noise=${encodeURIComponent(noise)}&spread=${encodeURIComponent(spread)}`);

export const getGlossary = () => jfetch('/api/glossary');

export const getToolAuthoring = () => jfetch('/api/tools/authoring');
