// Wait for the response before scheduling the next refresh. Completed runs
// load once; review saves and the Refresh evidence button request fresh data.
export function watchEvaluation(load, { poll, onError, interval = 15000 }) {
  const controller = new AbortController();
  let timer;
  async function refresh() {
    try { await load(controller.signal); }
    catch (failure) { if (!controller.signal.aborted) onError(failure); }
    if (!controller.signal.aborted && poll) timer = setTimeout(refresh, interval);
  }
  refresh();
  return () => { controller.abort(); clearTimeout(timer); };
}
