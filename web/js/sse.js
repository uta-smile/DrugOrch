/* The SSE client: reconnection, replay, and keeping a fast stream from killing the DOM.
 *
 * Two problems this solves that a bare EventSource does not.
 *
 * **Replay.** The server emits a monotonic `seq` as the SSE `id:`, so the browser's own
 * reconnect sends `Last-Event-ID` and resumes exactly where it stopped. We additionally pass
 * `from_seq` on the URL, because a page *reload* starts a fresh EventSource with no memory
 * of the header — and a reloaded page that silently skipped the first forty events would
 * render a trace with a hole in it that looks like a routing decision that never happened.
 *
 * **Backpressure.** SSE has no flow control and a docking round emits progress continuously.
 * Calling setState per event would re-render the whole trace hundreds of times a second. So
 * events land in a plain array and a single requestAnimationFrame loop drains it, producing
 * exactly one update per frame. Within a frame, progress events coalesce to their latest
 * value per subsystem; decisions and molecules never coalesce, because each is a discrete
 * fact the trace depends on.
 */

const COALESCE = new Set(['call_progress', 'policy_thinking', 'state']);
const TERMINAL = new Set(['run_finished', 'run_cancelled']);

// How often to drain while the tab is hidden. Nobody is watching, so this only has to keep
// the buffer from growing without bound and keep the view current enough that returning to
// the tab is instant.
const HIDDEN_DRAIN_MS = 250;

export class RunStream {
  /**
   * @param {string} runId
   * @param {(events: object[], meta: object) => void} onBatch  called once per frame
   * @param {(status: string) => void} onStatus  live | reconnecting | closed
   */
  constructor(runId, onBatch, onStatus) {
    this.runId = runId;
    this.onBatch = onBatch;
    this.onStatus = onStatus || (() => {});
    this.lastSeq = 0;
    this.buffer = [];
    this.dropped = 0;
    this.es = null;
    this.raf = null;
    this._onVisibility = null;
    this.closed = false;
    this.retries = 0;
    this.retryTimer = null;
  }

  start(fromSeq = 0) {
    this.lastSeq = fromSeq;
    this._open();
    this._pump();
  }

  _open() {
    if (this.closed) return;
    const url = `/api/runs/${encodeURIComponent(this.runId)}/events?from_seq=${this.lastSeq}`;
    const es = new EventSource(url);
    this.es = es;

    const handle = (ev) => {
      let data;
      try {
        data = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (data.seq <= this.lastSeq) return; // a replayed duplicate after a reconnect
      this.lastSeq = data.seq;
      this.retries = 0;
      this.buffer.push(data);
      if (TERMINAL.has(data.type)) {
        // The run is over; close explicitly or EventSource reconnects forever.
        this._closeSource();
        this.onStatus('closed');
      }
    };

    // Named events, so a dropped progress tick costs nothing to identify.
    for (const t of [
      'run_started', 'state', 'policy_thinking', 'decision', 'call_started',
      'call_progress', 'call_finished', 'molecules_added', 'best_updated',
      'budget_block', 'policy_fallback', 'subsystem_error', 'warning',
      'run_finished', 'run_cancelled',
    ]) {
      es.addEventListener(t, handle);
    }
    es.addEventListener('message', handle);

    es.onopen = () => this.onStatus('live');
    es.onerror = () => {
      if (this.closed || es.readyState === EventSource.CLOSED) {
        // EventSource's own backoff is not configurable, so drive it ourselves.
        this._closeSource();
        if (!this.closed) this._scheduleRetry();
        return;
      }
      this.onStatus('reconnecting');
    };
  }

  _scheduleRetry() {
    this.onStatus('reconnecting');
    const delay = Math.min(30000, 1000 * 2 ** this.retries);
    this.retries += 1;
    clearTimeout(this.retryTimer);
    this.retryTimer = setTimeout(() => this._open(), delay);
  }

  _pump() {
    // requestAnimationFrame is the right clock for a visible tab -- it coalesces to exactly
    // one update per painted frame. But a hidden tab never gets a frame, so an rAF-only loop
    // stops draining entirely: a run opened in a background tab accumulates every event in
    // memory and renders nothing at all, and a scientist who leaves a long campaign in
    // another tab comes back to a blank trace. So the loop switches clocks with visibility
    // and keeps draining on a timer while hidden -- slower, which is correct when nobody is
    // looking, but never stopped.
    const drain = () => {
      if (this.buffer.length) {
        const batch = this.buffer;
        this.buffer = [];
        this.onBatch(coalesce(batch), { lastSeq: this.lastSeq, dropped: this.dropped });
      }
    };

    const tick = () => {
      drain();
      if (this.closed) return;
      this.raf = this._visible()
        ? requestAnimationFrame(tick)
        : setTimeout(tick, HIDDEN_DRAIN_MS);
    };

    this._onVisibility = () => {
      // Switch clocks immediately rather than waiting out the timer, so returning to the
      // tab repaints at once instead of up to HIDDEN_DRAIN_MS later.
      this._clearTimer();
      if (!this.closed) tick();
    };
    document.addEventListener('visibilitychange', this._onVisibility);
    tick();
  }

  _visible() {
    return typeof document === 'undefined' || document.visibilityState !== 'hidden';
  }

  _clearTimer() {
    if (this.raf == null) return;
    cancelAnimationFrame(this.raf);
    clearTimeout(this.raf);
    this.raf = null;
  }

  _closeSource() {
    if (this.es) {
      this.es.close();
      this.es = null;
    }
  }

  close() {
    this.closed = true;
    this._closeSource();
    clearTimeout(this.retryTimer);
    this._clearTimer();
    if (this._onVisibility) {
      // Without this the listener outlives the stream, and navigating between runs leaves
      // one dead drain loop per visit still firing on every tab switch.
      document.removeEventListener('visibilitychange', this._onVisibility);
      this._onVisibility = null;
    }
  }
}

/** Within one frame, keep only the latest of each continuously-valued event. */
function coalesce(batch) {
  const out = [];
  const latest = new Map();
  for (const ev of batch) {
    if (!COALESCE.has(ev.type)) {
      out.push(ev);
      continue;
    }
    const key = `${ev.type}:${ev.data?.subsystem || ''}`;
    latest.set(key, ev);
  }
  for (const ev of latest.values()) out.push(ev);
  out.sort((a, b) => a.seq - b.seq);
  return out;
}
