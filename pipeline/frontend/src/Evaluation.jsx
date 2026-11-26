import { useCallback, useEffect, useState } from "react";
import "./Evaluation.css";
import { holdsActiveSlot } from "./runLifecycle.js";
import { watchEvaluation } from "./evaluationRefresh.js";

const API = (import.meta.env.DEV ? "" : (import.meta.env.VITE_API_URL || "")).replace(/\/$/, "");
const base = `${API}/api/t1/evaluation/runs/`;
const title = (value) => value?.replaceAll("_", " ") || "Unavailable";
const number = (value) => value == null ? "Unavailable" : `${Number(value).toFixed(2)} ms`;
const percent = (value) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;

function ReviewCard({ record, runId, onSaved, enlarge, eager }) {
  const prediction = record.reviewable_prediction;
  const [verdict, setVerdict] = useState(record.review.verdict);
  const [corrected, setCorrected] = useState(record.review.correct_label || "");
  const [saving, setSaving] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const [error, setError] = useState("");
  const [inspection, setInspection] = useState(record.inspection?.status || "UNINSPECTED");
  const [note, setNote] = useState(record.inspection?.note || "");
  async function save() {
    setSaving(true); setError("");
    try {
      const response = await fetch(`${base}${runId}/records/${record.id}/${prediction ? "review" : "inspect"}/`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(prediction ? { verdict, correct_label: corrected || null, revision: record.review.revision } : { status: inspection, note: note || null, revision: record.inspection?.revision || 0 }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      await onSaved();
    } catch (failure) { setError(failure.message || "Could not save review."); }
    finally { setSaving(false); }
  }
  const image = imageFailed ? null : record.crop_reference;
  return <article className="evaluation-card" data-evidence-id={record.id}>
    <div className="evaluation-image">{image ? <button onClick={() => enlarge(image)} aria-label={`Enlarge ${record.domain} crop for track ${record.track_id}`}><img src={`${API}${image}`} loading={eager ? "eager" : "lazy"} width="340" height="240" onError={() => setImageFailed(true)} alt={`${record.domain} inference evidence for track ${record.track_id}`} /></button> : <div className="evidence-unavailable">{record.crop_reference ? <>Image could not load<button type="button" onClick={() => setImageFailed(false)}>Retry crop</button></> : <>Crop unavailable<span>No image has been reconstructed.</span></>}</div>}</div>
    <div className="evaluation-card-body">
      <div className="evaluation-kicker">{title(record.domain)} · Track {record.track_id} · {prediction ? (record.kind === "attempt" ? `Attempt ${record.attempt_ordinal} prediction · separate from final accuracy` : "Prediction") : record.kind === "quality" ? "Quality only · no model prediction" : record.kind === "attempt" ? `Attempt ${record.attempt_ordinal} · inspect separately` : "Lifetime outcome · inspect separately"}</div>
      <h3 dir={record.domain === "plate" ? "ltr" : "auto"}>{prediction ? record.predicted_label : title(record.outcome)}</h3>
      <p>Source frame {record.source_frame_index ?? "unavailable"} · Engine frame {record.frame_index ?? "unavailable"} · Video {record.video_timestamp == null ? "timestamp unavailable" : `${Number(record.video_timestamp).toFixed(3)} s`} · {title(record.outcome)}</p>
      <p className="evidence-scope">{record.input_scope || (record.kind === "quality" ? "Observed and quality-scored; not a model inference." : "Retained producer evidence")}</p>
      <dl><div><dt>Native inference</dt><dd>{number(record.inference_time_ms)}</dd></div><div><dt>Quality gate</dt><dd>{number(record.quality_time_ms)}</dd></div><div><dt>Preprocessing</dt><dd>{number(record.preprocessing_time_ms)}</dd></div><div><dt>Queue wait</dt><dd>{number(record.queue_wait_ms)}</dd></div><div><dt>Total attempt</dt><dd>{number(record.attempt_time_ms)}</dd></div><div><dt>Quality decision</dt><dd>{record.quality && typeof record.quality === "object" ? `${record.quality.eligible === true ? "Eligible" : record.quality.eligible === false ? "Rejected" : "Not assessed"} · ${(record.quality.reasons || []).join(", ") || record.quality.score}` : record.quality != null ? `Score ${Number(record.quality).toFixed(3)}` : "Unavailable"}</dd></div></dl>
      {record.quality && <p>Size {record.quality.original_width ?? record.quality.width ?? "?"} × {record.quality.original_height ?? record.quality.height ?? "?"} · Sharpness {record.quality.sharpness_variance ?? "unavailable"} · Dark {record.quality.dark_fraction ?? "?"} · Bright {record.quality.bright_fraction ?? "?"} · Border edges {record.quality.border_edges ?? "?"}</p>}
      {record.processing_incomplete && <p>Processing ended incomplete; retained predictions and attempts are preserved.</p>}
      {record.scheduling_reason && <p>Scheduling: {title(record.scheduling_reason)}</p>}
      {record.kind === "attempt" && <p>Returned manufacturer: {record.predicted_label || "unavailable"} · Native entered: {record.native_call_entered ? "yes" : "no"} · {record.error || ""}</p>}
      {!prediction && record.review.revision > 0 && record.review.verdict !== "UNREVIEWED" && <p>Earlier review: {record.review.verdict} · Preserved separately; excluded from prediction accuracy.</p>}
      {record.historical_producer_duration_ms != null && <p>Historical producer duration: {number(record.historical_producer_duration_ms)} · Native timing unavailable.</p>}
      <details><summary>Model, inputs & provenance</summary><p className="model-version">{record.model_version || "Model provenance unavailable for this historical record."}</p><p>{record.quality_policy_version || "Quality policy unavailable"}</p><p className="model-version">Monotonic start: {record.inference_start_ns || "unavailable"} ns<br />Completion: {record.inference_completion_ns || "unavailable"} ns</p>{record.load_time_ms != null && <p>Model loading: {number(record.load_time_ms)}</p>}{record.queue_wait_ms != null && <p>Queue wait: {number(record.queue_wait_ms)}</p>}{record.original_frame_reference && <button className="evidence-link" onClick={() => enlarge(record.original_frame_reference)}>Inspect original detector frame</button>}{record.supporting_crops?.map((url, i) => <button className="evidence-link" key={url} onClick={() => enlarge(url)}>Inference input {i + 1}</button>)}</details>
      {prediction ? <><fieldset className="verdict-options"><legend>Human verdict <span>{record.review.verdict}</span></legend>{["TRUE", "FALSE", "NOT CLEAR", "UNREVIEWED"].map((value) => <label key={value} className={verdict === value ? "chosen" : ""}><input type="radio" name={`verdict-${record.id}`} value={value} checked={verdict === value} onChange={() => setVerdict(value)} />{value}</label>)}</fieldset>
      {verdict === "FALSE" && <label className="correct-label">Correct label · optional<input aria-label={`Correct label for ${record.id}`} value={corrected} maxLength={2048} onChange={(event) => setCorrected(event.target.value)} dir={record.domain === "plate" ? "ltr" : "auto"} /></label>}</> : <><fieldset className="verdict-options"><legend>Inspection · excluded from prediction accuracy</legend>{["UNINSPECTED", "INSPECTED", "NEEDS_ATTENTION"].map((value) => <label key={value}><input type="radio" name={`inspection-${record.id}`} checked={inspection === value} onChange={() => setInspection(value)} />{title(value)}</label>)}</fieldset><label>Inspection note<input aria-label={`Inspection note for ${record.id}`} value={note} maxLength={2048} onChange={(e) => setNote(e.target.value)} /></label></>}
      <div className="review-save"><button disabled={saving} onClick={save}>{saving ? "Saving…" : prediction ? "Save review" : "Save inspection"}</button><small>Revision {prediction ? record.review.revision : record.inspection?.revision || 0}</small></div>
      {error && <p role="alert" className="review-error">{error}</p>}
    </div>
  </article>;
}

export default function Evaluation({ initialRunId = "" }) {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState(() => initialRunId || localStorage.getItem("waycon-evaluation-run") || "");
  const [data, setData] = useState(null);
  const [model, setModel] = useState("");
  const [filter, setFilter] = useState("all");
  const [domain, setDomain] = useState("");
  const [view, setView] = useState("prediction");
  const [pagination, setPagination] = useState({ key: "", page: 0 });
  const [image, setImage] = useState(null);
  const [error, setError] = useState("");
  const refresh = useCallback(async (signal) => {
    if (!runId) return;
    const response = await fetch(`${base}${runId}/`, { signal, cache: "no-store" });
    if (!response.ok) throw new Error("Could not load evaluation evidence.");
    const body = await response.json();
    if (!signal?.aborted) setData(body);
  }, [runId]);
  useEffect(() => {
    const controller = new AbortController();
    fetch(base, { signal: controller.signal, cache: "no-store" }).then(async (response) => {
      if (!response.ok) throw new Error("Evaluation API unavailable.");
      setRuns((await response.json()).runs);
    }).catch((failure) => { if (failure.name !== "AbortError") setError(failure.message); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!runId || !runs.length) return;
    return watchEvaluation(refresh, {
      poll: holdsActiveSlot(runs.find((r) => r.run_id === runId)),
      onError: (failure) => setError(failure.message),
    });
  }, [runId, refresh, runs]);
  useEffect(() => {
    if (!image) return;
    const escape = (event) => { if (event.key === "Escape") setImage(null); };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [image]);
  const activeData = data?.run_id === runId ? data : null;
  const records = (activeData?.records || []).filter((r) => !model || r.model_version === model);
  const calls = (activeData?.calls || []).filter((c) => !model || c.model_version === model);
  const finalized = new Set(records.filter((r) => r.kind === "outcome").map((r) => r.lifetime_id));
  const visible = records.filter((r) => {
    const kind = r.display_kind || r.kind;
    const category = ["outcome", "lifetime"].includes(kind) ? "observation" : kind;
    if (kind === "lifetime" && finalized.has(r.lifetime_id)) return false;
    const status = r.reviewable_prediction ? r.review.verdict : r.inspection?.status;
    return category === view && (!domain || r.domain === domain) &&
      (filter === "all" || (filter === "reviewed" ? status !== "UNREVIEWED" && status !== "UNINSPECTED" : status === filter));
  });
  const paginationKey = [runId, model, domain, view, filter].join("|");
  const pages = Math.max(1, Math.ceil(visible.length / 48));
  const pageNumber = pagination.key === paginationKey ? Math.min(pagination.page, pages - 1) : 0;
  const pageRecords = visible.slice(pageNumber * 48, (pageNumber + 1) * 48);
  const models = [...new Set([...(activeData?.records || []).map((r) => r.model_version), ...(activeData?.calls || []).map((c) => c.model_version)].filter(Boolean))];
  // Server is the accuracy and timing authority, including filtered-model summaries.
  const [summary, setSummary] = useState(null);
  useEffect(() => {
    if (!runId || !model || data?.run_id !== runId) return;
    const controller = new AbortController();
    fetch(`${base}${runId}/?model=${encodeURIComponent(model)}`, { signal: controller.signal, cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error("Could not load model metrics.");
        return r.json();
      }).then((body) => {
        if (!controller.signal.aborted) setSummary({ runId, model, metrics: body.metrics });
      }).catch((failure) => { if (!controller.signal.aborted) setError(failure.message); });
    return () => controller.abort();
  }, [runId, model, data]);
  const displayedSummary = model ? (summary?.runId === runId && summary?.model === model ? summary.metrics : null) : activeData?.metrics;
  const reviewed = records.filter((r) => r.accuracy_prediction && r.review.verdict !== "UNREVIEWED").length;
  const total = records.filter((r) => r.accuracy_prediction).length;
  return <section className="evaluation-workspace" aria-label="Human evaluation">
    <div className="evaluation-heading"><div><div className="eyebrow">EVIDENCE / HUMAN VERIFICATION</div><h2>Evaluation & review</h2><p>Ground truth stays separate from production predictions.</p></div><a className={`csv-download ${runId ? "" : "disabled"}`} href={runId ? `${base}${runId}/download.csv?model=${encodeURIComponent(model)}` : undefined}>Download CSV · TRUE + FALSE</a></div>
    <div className="evaluation-toolbar"><label>Video-analysis run<select aria-label="Evaluation run" value={runId} onChange={(event) => { setRunId(event.target.value); localStorage.setItem("waycon-evaluation-run",event.target.value); setModel(""); setData(null); setError(""); }}><option value="">Select a run</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.filename ? `${run.filename} · ` : ""}{run.run_id}{run.historical ? " · historical evidence gaps" : ""}</option>)}</select></label><label>Model<select aria-label="Evaluation model" value={model} onChange={(e) => setModel(e.target.value)}><option value="">All models · separate task metrics</option>{models.map((m) => <option key={m}>{m}</option>)}</select></label></div>
    {runId && <button type="button" className="evidence-link" onClick={() => refresh().catch((failure) => setError(failure.message))}>Refresh evidence</button>}
    {error && <p role="alert" className="error-banner">{error}</p>}
    {runId && <><div className="evaluation-progress"><strong>{reviewed} / {total} predictions reviewed</strong><progress max={Math.max(1,total)} value={reviewed} /><span>{displayedSummary?.total_model_calls ?? 0} model calls · {displayedSummary?.model_failures ?? 0} failures · {displayedSummary?.model_timeouts ?? 0} timeouts · {calls.length} total measured events</span></div>
      <div className="panel evaluation-completeness" aria-label="Manufacturer completeness"><strong>Manufacturer lifetimes: {displayedSummary?.manufacturer_completeness?.lifetimes ?? 0} · Final outcomes: {displayedSummary?.manufacturer_completeness?.finalized ?? 0} · Pending: {displayedSummary?.manufacturer_completeness?.pending ?? 0}</strong><p>{Object.entries(displayedSummary?.manufacturer_completeness?.final_outcomes || {}).map(([key,count]) => `${title(key)}: ${count}`).join(" · ") || "Historical completeness was not instrumented."}</p><p>Unknowns, quality rejections, failures and inspections are excluded from prediction accuracy. Attempts are separate from final predictions.</p></div>
      <div className="table-scroll panel"><table className="evaluation-metrics"><caption>Accuracy by domain · denominator TRUE + FALSE; unclear and unreviewed excluded</caption><thead><tr><th>Domain</th><th>Predictions</th><th>TRUE</th><th>FALSE</th><th>NOT CLEAR</th><th>Unreviewed</th><th>Accuracy</th><th>Coverage</th><th>Unclear rate</th></tr></thead><tbody>{Object.entries(displayedSummary?.domains || {}).map(([name,m]) => <tr key={name}><th>{title(name)}{name === "plate" && " · exact full text"}</th><td>{m.total}</td><td>{m.correct}</td><td>{m.incorrect}</td><td>{m.unclear}</td><td>{m.unreviewed}</td><td>{percent(m.evaluated_accuracy)}</td><td>{percent(m.review_coverage)}</td><td>{percent(m.unclear_rate)}</td></tr>)}</tbody></table></div>
      <p>Plate evidence: {displayedSummary?.plate_outcomes?.reads ?? 0} reads · {displayedSummary?.plate_outcomes?.unreadable ?? 0} unreadable · {displayedSummary?.plate_outcomes?.no_detection_observations ?? 0} no-detection observations · {displayedSummary?.plate_outcomes?.human_confirmed_detection_failures ?? 0} human-marked missed detections. Category correctness is a separate task.</p>
      <details className="panel evaluation-latency" open><summary>Execution timing · mean, median, p95, p99 and Observed WCET</summary><p>Successful calls are summarized below. Failures and timeouts are counted separately. Observed WCET is the maximum observed latency, not a formal hard real-time guarantee.</p><div className="table-scroll"><table><thead><tr><th>Stage</th><th>Calls / measured successes</th><th>Mean</th><th>Median</th><th>p95</th><th>p99</th><th>Min</th><th>Observed WCET</th><th>Std dev</th><th>Failures</th><th>Timeouts</th></tr></thead><tbody>{Object.entries(displayedSummary?.stages || {}).map(([stage,m]) => <tr key={stage}><th>{title(stage)}<small>{(m.scopes || []).join(" ")}</small></th><td>{m.total_calls} / {m.count}</td><td>{number(m.mean)}</td><td>{number(m.median)}</td><td>{number(m.p95)}</td><td>{number(m.p99)}</td><td>{number(m.minimum)}</td><td>{number(m.observed_wcet)}</td><td>{number(m.standard_deviation)}</td><td>{m.failures}</td><td>{m.timeouts}</td></tr>)}</tbody></table></div>{!calls.length && <p>No measured calls for this selection. Historical timings are never inferred from video duration.</p>}</details>
      <div className="evaluation-toolbar"><div className="evaluation-tabs" role="tablist">{[["prediction","Predictions"],["attempt","Model attempts"],["quality","Quality review"],["observation","Lifetime outcomes / missing / failed"]].map(([key,text]) => <button key={key} role="tab" aria-selected={view === key} onClick={() => { setView(key); setFilter("all"); }}>{text}</button>)}</div><label>Task<select value={domain} onChange={(e) => setDomain(e.target.value)}><option value="">All domains</option>{["manufacturer","plate","plate_category","type","colour"].map((d) => <option key={d}>{d}</option>)}</select></label><label>Review filter<select aria-label="Review filter" value={filter} onChange={(e) => setFilter(e.target.value)}>{(["prediction", "attempt"].includes(view) ? [["all","All"],["reviewed","Reviewed"],["UNREVIEWED","Unreviewed"],["FALSE","False"],["NOT CLEAR","Unclear"],["UNINSPECTED","Uninspected"],["INSPECTED","Inspected"],["NEEDS_ATTENTION","Needs attention"]] : [["all","All"],["UNINSPECTED","Uninspected"],["INSPECTED","Inspected"],["NEEDS_ATTENTION","Needs attention"]]).map(([value,text]) => <option key={value} value={value}>{text}</option>)}</select></label></div>
      {view === "quality" && <p>Every quality observation is separate from model predictions. Representative rejected crops are retained where captured; other observations show image unavailable. Quality inspections never enter prediction accuracy or delete an image.</p>}
      <div className="evaluation-grid">{pageRecords.map((r, index) => <ReviewCard key={`${r.id}-${r.review.revision}-${r.inspection?.revision || 0}`} record={r} runId={runId} onSaved={refresh} enlarge={setImage} eager={index < 3} />)}</div>{!visible.length && <div className="evaluation-empty">No evidence matches these filters. New instrumented runs retain actual inputs; older runs may have no crops.</div>}
      <nav aria-label="Evidence pages"><button disabled={pageNumber === 0} onClick={() => setPagination({ key: paginationKey, page: pageNumber - 1 })}>Previous</button> <span>Page {pageNumber + 1} / {pages} · {visible.length} records</span> <button disabled={pageNumber + 1 === pages} onClick={() => setPagination({ key: paginationKey, page: pageNumber + 1 })}>Next</button></nav>
      <a className="evidence-link" href={`${base}${runId}/audit/`} target="_blank" rel="noreferrer">Inspect review audit trail</a>
    </>}
    {!runId && <div className="evaluation-empty">Select an existing recorded-video run to inspect predictions, crops and timing.</div>}
    {image && <div className="image-inspector" role="dialog" aria-modal="true" aria-label="Enlarged evidence image" onClick={() => setImage(null)}><button autoFocus onClick={() => setImage(null)} aria-label="Close image inspector">Close ×</button><img src={`${API}${image}`} alt="Enlarged inference evidence" onClick={(event) => event.stopPropagation()} /></div>}
  </section>;
}
