import { useEffect, useState } from "react";
import "./App.css";
import { cleanupLabel, holdsActiveSlot } from "./runLifecycle.js";
import { cropPath, plateReasonLabel, sourceFrame, cropOrigin, brandName } from "./cropEvidence.js";
import { samplingDefaults, validInterval, samplingLabel } from "./frameSampling.js";
import Evaluation from "./Evaluation.jsx";

// The dev server proxies API requests so one SSH forward serves the console.
// Production builds retain the configured deployment API origin.
const API = (import.meta.env.DEV ? "" : (import.meta.env.VITE_API_URL || "")).replace(/\/$/, "");
const endpoint = (path) => `${API}/api/t1/runs/${path}`;
const label = (value) => value ? value.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()) : "Unknown";
const deliveryLabel = { not_configured: "Not configured", not_attempted: "Not attempted", acknowledged: "Acknowledged", committed: "Committed", no_deliveries: "No deliveries", pending: "Pending", disabled: "Disabled" };

function Brand({ result, deliveryReason }) {
  const name = brandName(result);
  return <div className="brand-result"><strong>{name}</strong>{deliveryReason === "catalogue_unmapped" && <small>No approved DB1 mapping · review available</small>}{deliveryReason === "mapped" && <small>DB1 catalogue matched</small>}</div>;
}

function TrackCrop({ runId, trackId, kind, crop }) {
  const path = cropPath(runId, trackId, kind, crop);
  const [failedPath, setFailedPath] = useState(null);
  const available = path && failedPath !== path;
  const origin = cropOrigin(kind, crop);
  return <div className={`track-crop ${kind}`}>
    {available ? <img src={`${API}${path}`} alt={`${origin} crop for vehicle track ${trackId}`} loading="lazy" onError={() => setFailedPath(path)} /> : <span className="crop-missing">Crop unavailable</span>}
    {available && <small>{origin}</small>}<small>{sourceFrame(crop?.frame)}</small>
  </div>;
}

function PlateCrop({ row }) {
  const plate = row.plate;
  if (cropPath(row.run_id, row.track_id, "plate", plate.crop)) return <>{plate.status !== "read" && <small>{plateReasonLabel(plate)}</small>}<TrackCrop runId={row.run_id} trackId={row.track_id} kind="plate" crop={plate.crop} /></>;
  return <div className="plate-crop-evidence"><span>{plateReasonLabel(plate)}</span><small>{sourceFrame(plate.diagnostic?.frame)}</small></div>;
}

function App() {
  const [view, setView] = useState("console");
  const [kind, setKind] = useState("mp4");
  const [file, setFile] = useState(null);
  const [uploaded, setUploaded] = useState(null);
  const [videoPath, setVideoPath] = useState("");
  const [uploadPercent, setUploadPercent] = useState(0);
  const [uploadState, setUploadState] = useState("");
  const [uploadLimits, setUploadLimits] = useState(null);
  const [reviewRun, setReviewRun] = useState("");
  const [uri, setUri] = useState("");
  const [transport, setTransport] = useState("tcp");
  const [manufacturer, setManufacturer] = useState(true);
  const [intervals, setIntervals] = useState(samplingDefaults);
  const sampleEvery = intervals[kind];
  const [maxSeconds, setMaxSeconds] = useState("");
  const [runs, setRuns] = useState([]);
  const [observedAt, setObservedAt] = useState(() => Date.now());
  const [selected, setSelected] = useState("");
  const [backend, setBackend] = useState("checking");
  const [stack, setStack] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [previewFailed, setPreviewFailed] = useState(false);
  const run = runs.find((r) => r.run_id === selected) || (!selected ? runs[0] : null);
  const active = runs.find(holdsActiveSlot);

  useEffect(() => {
    const controller = new AbortController();
    let alive = true;
    async function poll() {
      try {
        const response = await fetch(endpoint(""), { signal: controller.signal, cache: "no-store" });
        if (!response.ok) throw new Error("backend unavailable");
        const data = await response.json();
        const stackResponse = await fetch(`${API}/api/t1/stack/`, { signal: controller.signal, cache: "no-store" });
        const stackData = stackResponse.ok ? await stackResponse.json() : null;
        if (alive) { setRuns(data.runs); setObservedAt(Date.now()); setBackend("online"); setStack(stackData); }
      } catch { if (alive) setBackend("offline"); }
    }
    poll();
    const timer = setInterval(poll, 900);
    return () => { alive = false; controller.abort(); clearInterval(timer); };
  }, []);

  useEffect(() => {
    fetch(`${API}/api/t1/uploads/`, { cache: "no-store" }).then((response) => response.json()).then(setUploadLimits).catch(() => {});
  }, []);

  function uploadVideo() {
    if (!file) return;
    if (uploadLimits?.max_bytes && file.size > uploadLimits.max_bytes) return setError("Video exceeds the server upload limit.");
    setError(""); setUploaded(null); setUploadState("Uploading"); setUploadPercent(0);
    const request = new XMLHttpRequest();
    request.open("POST", `${API}/api/t1/uploads/`);
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.setRequestHeader("X-Video-Filename", encodeURIComponent(file.name));
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) setUploadPercent(Math.round(100 * event.loaded / event.total));
      if (event.loaded === event.total) setUploadState("Validating video contents…");
    };
    request.onload = () => {
      let data;
      try { data = JSON.parse(request.responseText); } catch { data = {}; }
      if (request.status === 201) { setUploaded(data); setUploadState("Validated · Ready for analysis"); setUploadPercent(100); }
      else { setUploadState("Upload rejected"); setError(data.error || "Upload failed."); }
    };
    request.onerror = () => { setUploadState("Upload failed"); setError("Upload connection lost. Select Upload Video to retry."); };
    request.send(file);
  }
  const uploading = uploadState === "Uploading" || uploadState.startsWith("Validating");

  async function start(event) {
    event.preventDefault();
    if (kind === "mp4" && !uploaded && !videoPath.trim()) return setError("Upload a video or enter its absolute path on the server.");
    if (kind === "rtsp" && !uri) return setError("Enter the camera URI for this Start.");
    if (!validInterval(sampleEvery)) return setError("Analyze every Nth frame must be an integer of at least 1.");
    setBusy(true); setError(""); setPreviewFailed(false);
    let body;
    const headers = {};
    if (kind === "mp4") {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify({ source_kind: kind, ...(videoPath.trim() ? { video_path: videoPath.trim() } : { upload_id: uploaded.upload_id }),
        manufacturer_enabled: manufacturer, sample_every: Number(sampleEvery),
        max_seconds: maxSeconds === "" ? null : Number(maxSeconds) });
    } else {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify({ source_kind: kind, rtsp_uri: uri, rtsp_transport: transport, manufacturer_enabled: manufacturer, sample_every: Number(sampleEvery), max_seconds: maxSeconds === "" ? null : Number(maxSeconds) });
    }
    // The URI is transient, body-only, and is never persisted in browser storage.
    setUri("");
    try {
      const response = await fetch(endpoint(""), { method: "POST", headers, body });
      let data;
      try { data = await response.json(); }
      catch { throw new Error(`Start received an empty or invalid backend response (HTTP ${response.status}). Check backend/proxy logs and run history before retrying; the run may already have started.`); }
      if (!response.ok) throw new Error(data.error || "Could not start the run.");
      setRuns((previous) => [data, ...previous.filter((r) => r.run_id !== data.run_id)]);
      setSelected(data.run_id);
    } catch (failure) { setError(failure.message || "Could not start. Check the input, backend connection, and run limits."); }
    finally { setBusy(false); }
  }
  async function stop() {
    if (!run) return;
    setBusy(true); setError("");
    try {
      const response = await fetch(endpoint(`${run.run_id}/stop/`), { method: "POST" });
      if (!response.ok) throw new Error("stop failed");
      const data = await response.json();
      setRuns((previous) => previous.map((r) => r.run_id === data.run_id ? data : r));
    } catch { setError("Could not request Stop. The run status will continue updating."); }
    finally { setBusy(false); }
  }
  const rows = run?.rows || [];
  const running = holdsActiveSlot(run);
  const cleanup = cleanupLabel(run);
  return <div className="console-shell">
    <aside className="sidebar"><a className="wordmark" href="#main"><img className="brand-logo" src="/any-vehicle-logo.svg" alt="" /><span>Any Vehicle<small>VEHICLE INTELLIGENCE</small></span></a>
      <div className="workspace-label">YOUR WORKSPACE</div><button className={`nav-item ${view === "console" ? "active" : ""}`} onClick={() => setView("console")}><span>▤</span> Vehicle Console</button><button className={`nav-item ${view === "evaluation" ? "active" : ""}`} onClick={() => setView("evaluation")}><span>✓</span> Evaluation / Review</button>
      <div className="sidebar-note"><span className="sidebar-note-label">A CLEARER PICTURE</span><p>From a passing vehicle<br />to meaningful intelligence.</p><div className="sidebar-note-lines" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /></div></div>
      <div className="backend-state"><span className={`signal-dot ${backend}`} /><span>Backend {backend}</span><small>Local processing</small></div>
    </aside>
    <main id="main"><header className="page-header"><div><div className="eyebrow">VEHICLE INTELLIGENCE, IN FOCUS</div><h1>Any Vehicle<span>.</span></h1><p>Every frame tells a story. Bring yours into focus.</p></div><div className="header-status"><span className="environment"><span className={`signal-dot ${backend}`} />{backend === "online" ? "Backend connected" : backend === "offline" ? "Backend offline" : "Connecting…"}</span><span className="header-caption">VIDEO ANALYTICS WORKSPACE</span></div></header>
      {view === "evaluation" ? <Evaluation initialRunId={reviewRun} /> : <><div className={stack?.full_stack_ready ? "run-identity" : "error-banner"} role="status" aria-label="Stack readiness">{stack?.full_stack_ready ? "Full stack ready · PostgreSQL test database · Vehicle attribute delivery enabled" : (stack?.error || "Checking full stack readiness…")}</div>
      <section className="insight-bar" aria-label="Selected run overview"><div><span className="insight-icon" aria-hidden="true">↗</span><div><span className="insight-label">VEHICLES</span><strong>{rows.length}<small>joined results</small></strong></div></div><div><span className="insight-icon" aria-hidden="true">▣</span><div><span className="insight-label">PLATE READS</span><strong>{rows.filter((row) => row.plate.status === "read").length}<small>accepted readings</small></strong></div></div><div><span className="insight-icon" aria-hidden="true">◷</span><div><span className="insight-label">FRAMES ANALYZED</span><strong>{(run?.analyzed_frames ?? run?.frames ?? 0).toLocaleString()}<small>selected run</small></strong></div></div><div className="insight-source"><span className="insight-label">INPUT SOURCE</span><strong>{run ? run.source_kind === "rtsp" ? "RTSP camera" : "Recorded video" : "Ready to connect"}</strong><span>{run ? label(run.status) : "Upload a video or connect a camera"}</span></div></section>
      <div className="console-layout"><section className="panel source-panel"><div className="panel-title"><span className="step">01</span><h2>Input & controls</h2></div>
        <form onSubmit={start}><div className="source-tabs" role="tablist" aria-label="Input source"><button type="button" role="tab" aria-selected={kind === "mp4"} className={kind === "mp4" ? "selected" : ""} disabled={!!active} onClick={() => setKind("mp4")}>Upload Video</button><button type="button" role="tab" aria-selected={kind === "rtsp"} className={kind === "rtsp" ? "selected" : ""} disabled={!!active} onClick={() => setKind("rtsp")}>RTSP camera</button></div>
          {kind === "mp4" ? <div className="upload-video"><label className="file-picker"><span className="file-symbol">↥</span><strong>{file?.name || "Choose a video"}</strong><span>{file ? `${(file.size / 1048576).toFixed(1)} MB` : "MP4, MOV, AVI, MKV, WebM · Server decoder validates contents"}</span><input aria-label="Video file" type="file" accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v,.mpeg,.mpg,.ts,.mts,.wmv,.flv,.3gp" disabled={uploading} onChange={(e) => { setFile(e.target.files[0] || null); setUploaded(null); setUploadState(""); setUploadPercent(0); }} /></label>
            <button type="button" className="upload-button" onClick={uploadVideo} disabled={!file || uploading}>{uploading ? "Uploading / validating…" : "Upload Video"}</button>
            <label>Or use a video already on the server<input aria-label="Server video path" type="text" placeholder="/home/intern/part-time/videos/example.mp4" value={videoPath} disabled={uploading || busy || !!active} onChange={(e) => setVideoPath(e.target.value)} /></label>
            {videoPath.trim() && <p className="bounded-note">Start Analysis reads this server file directly. Keep it available and unchanged until the run finishes.</p>}
            {uploadState && <div className="upload-status" role="status"><progress aria-label="Upload progress" max="100" value={uploadPercent} /><span>{uploadPercent}% · {uploadState}</span></div>}
            {uploaded && <p className="bounded-note">{uploaded.container} · {uploaded.width} × {uploaded.height} · {uploaded.total_frames ?? "Unknown"} frames{uploaded.frame_count_exact === false ? " (estimated)" : ""}. Saved independently of this browser.</p>}
            {uploadLimits?.max_bytes && <p className="bounded-note">Upload limit: {(uploadLimits.max_bytes / 1073741824).toFixed(1)} GiB. Videos and results are retained.</p>}
          </div> : <div className="camera-input"><label htmlFor="camera-uri">Camera URI</label><input id="camera-uri" aria-label="RTSP URI" required type="password" autoComplete="new-password" spellCheck={false} placeholder="Enter URI for this Start" disabled={!!active} value={uri} onChange={(e) => setUri(e.target.value)} /><p>Sent privately to the backend. Cleared on Start. Enter a new URI for each run.</p><label htmlFor="rtsp-transport">RTSP transport</label><select id="rtsp-transport" aria-label="RTSP transport" value={transport} disabled={!!active} onChange={(e) => setTransport(e.target.value)}><option value="tcp">TCP</option><option value="udp">UDP</option></select></div>}
          <div className="run-limits"><label>Analyze every Nth frame<input aria-label="Analyze every Nth frame" type="number" min="1" step="1" required value={sampleEvery} disabled={!!active} onChange={(e) => setIntervals({ ...intervals, [kind]: e.target.value })} /></label><label>Timeout · optional seconds<input aria-label="Timeout" type="number" min="1" step="1" placeholder="No timeout" value={maxSeconds} disabled={!!active} onChange={(e) => setMaxSeconds(e.target.value)} /></label></div>
          <div className="recognition-option"><div><label htmlFor="manufacturer-toggle">Vehicle attributes</label><small>Local LFM · manufacturer, body type and colour</small></div><input id="manufacturer-toggle" type="checkbox" role="switch" checked={manufacturer} disabled={!!active} onChange={(e) => setManufacturer(e.target.checked)} /></div>
          <div className="action-row"><button className="start-button" type="submit" disabled={busy || uploading || !!active || backend !== "online" || (stack?.mode === "fixture" && !stack.full_stack_ready) || (kind === "mp4" && !uploaded && !videoPath.trim())}>▶ Start Analysis</button><button className="stop-button" type="button" onClick={stop} disabled={busy || !running}>■ Stop</button></div>
          <p className="bounded-note">Videos run through EOF with every frame by default. Stop or an optional timeout ends a partial run.</p>
        </form>
      </section>
      <section className="panel preview-panel"><div className="panel-title"><span className="step">02</span><h2>Scene overview</h2><span className={`run-badge ${run?.status || "idle"}`}>{label(run?.status || "idle")}</span></div>
        <div className="preview-stage">{run?.preview_revision > 0 && !previewFailed ? <img key={run.run_id} src={endpoint(`${run.run_id}/preview/?revision=${run.preview_revision}`)} alt={`Backend frame preview for run ${run.run_id}`} onError={() => setPreviewFailed(true)} /> : <div className="preview-empty"><div className="scene-illustration" aria-hidden="true"><div className="scan-corner tl" /><div className="scan-corner tr" /><div className="scan-corner bl" /><div className="scan-corner br" /><svg viewBox="0 0 200 120" fill="none"><path d="m49 58 13-28h76l13 28" stroke="currentColor" strokeWidth="3" strokeLinejoin="round"/><path d="M43 57h114a8 8 0 0 1 8 8v27H35V65a8 8 0 0 1 8-8Z" stroke="currentColor" strokeWidth="3"/><path d="M47 92v10m106-10v10M53 72h20m54 0h20M88 81h24" stroke="currentColor" strokeWidth="5" strokeLinecap="round"/><path d="M79 30 68 57m53-27 11 27" stroke="currentColor" strokeWidth="2" opacity=".4"/></svg><div className="scan-line" /></div><strong>{running ? "Preparing your feed" : "Ready when you are"}</strong><p>{running ? run.phase : "Connect your source and watch vehicle intelligence come to life."}</p></div>}<div className="preview-caption"><span>{run ? `${run.source_kind === "mp4" ? "VIDEO" : run.source_kind.toUpperCase()} · ${run.source_kind === "rtsp" ? "Live source frames" : "Backend frames"}` : "NO ACTIVE INPUT"}</span><span>{run ? `${run.decoded_frames ?? run.frames} decoded · ${run.analyzed_frames ?? run.frames} analyzed` : "—"}</span></div></div>
        <div className="run-progress" aria-live="polite"><span className={running ? "activity-dot" : ""} />{run?.phase || "Choose an input to begin."}</div>
      </section></div>
      {run && <div className="run-identity" aria-label="Frame counters"><span>Decoded: {run.decoded_frames ?? run.frames}</span><span>Analyzed: {run.analyzed_frames ?? run.frames} / {run.total_frames ?? "unknown"} frames{run.upload?.frame_count_exact === false ? " (estimated total)" : ""}</span><span>Elapsed: {Math.max(0, Math.floor((new Date(run.ended_at || observedAt) - new Date(run.started_at)) / 1000))} s</span><span>Sampling skips: {run.sampling_skips ?? "Unknown (historical)"}</span>{run.source_kind === "rtsp" && <span>Overload drops: {run.overload_drops ?? "Unknown (historical)"}</span>}<span>{run.sample_every ? `Every ${run.sample_every} frame(s)` : "Historical frame-limit run"}</span><span>{run.end_reason ? `${run.full_video_complete ? "Full video complete" : "Partial / stream ended"} · ${label(run.end_reason)}` : ""}</span></div>}
      {error && <div className="error-banner" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError("")}>×</button></div>}
      {run?.error_code && <div className="error-banner" role="alert">{run.error_code === "input_unavailable" ? "The input could not be opened. Check the camera connection and start a new run." : "The backend could not finish this run. Its status and completed results are retained."}</div>}
      {run?.error_code === "db1_delivery_failed" && <div className="error-banner" role="alert">DB1 delivery failed. The full stack is not ready. Commit and ACK states show the last confirmed boundary.</div>}
      <section className="delivery-strip" aria-label="Delivery states"><div><span>POSTGRESQL COMMIT</span><strong>{deliveryLabel[run?.delivery.postgres_commit] || "Not configured"}</strong></div><div><span>REDIS DB1 ACK</span><strong>{deliveryLabel[run?.delivery.redis_ack] || "Not attempted"}</strong></div><div><span>ATTRIBUTE RESULT HANDOFF</span><strong>{run ? deliveryLabel[run.delivery.review_ack] : "—"}</strong></div><p>Approved manufacturer labels join vehicle, colour and plate results in DB1. Unmapped labels stay available for review.</p></section>
      <section className="panel results-panel"><div className="results-heading"><div><div className="eyebrow">JOINED BY VEHICLE TRACK</div><h2>Vehicle activity <span className="count">{rows.length}</span></h2></div><label className="run-select">Run history<select aria-label="Run history" value={run?.run_id || ""} onChange={(e) => { setSelected(e.target.value); setPreviewFailed(false); }}><option value="" disabled>No runs yet</option>{runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.source_kind === "mp4" ? "VIDEO" : r.source_kind.toUpperCase()} · {r.upload?.filename || r.run_id.slice(0,8)} · {label(r.status)} · {samplingLabel(r)}</option>)}</select></label></div>
        {run && <div className="run-identity"><button className="open-evaluation" onClick={() => { setReviewRun(run.run_id); localStorage.setItem("waycon-evaluation-run",run.run_id); setView("evaluation"); }}>Open evaluation results</button><span>{run.upload?.filename}</span><span>RUN</span><code>{run.run_id}</code><span>·</span><span>{new Date(run.started_at).toLocaleString()}</span><span>· Vehicle attributes {run.manufacturer_enabled ? "on" : "off"} for this run</span></div>}
        <div className="table-scroll"><table><thead><tr><th>Vehicle track</th><th>Type</th><th>Colour</th><th>Plate decision</th><th>Manufacturer</th><th>PostgreSQL commit</th><th>Redis DB1 ACK</th></tr></thead><tbody>{rows.map((row) => <tr key={`${row.run_id}:${row.track_id}`} data-track-id={row.track_id}><td><strong className="track-id">#{row.track_id}</strong><small>Frames {row.first_frame}–{row.last_frame}</small><TrackCrop runId={row.run_id} trackId={row.track_id} kind="vehicle" crop={row.vehicle_crop} /></td><td>{label(row.vehicle_type)}</td><td><span className="colour-label"><i style={{ background: row.colour.label || "transparent" }} />{label(row.colour.label)}</span><small>{row.colour.status === "undecided" ? "Undecided" : ""}</small></td><td><span className={`decision ${row.plate.status}`}>{label(row.plate.status)}</span>{row.plate.status === "read" && row.plate.text && <div><bdi dir="auto">{row.plate.text}</bdi></div>}{row.plate.category && <small>{label(row.plate.category)}</small>}<PlateCrop row={row} /></td><td><Brand result={row.manufacturer} deliveryReason={row.delivery.manufacturer_db1} /></td><td className="delivery-cell">{deliveryLabel[row.delivery.postgres_commit]}</td><td className="delivery-cell">{deliveryLabel[row.delivery.redis_ack]}</td></tr>)}</tbody></table>{!rows.length && <div className="results-empty"><span>◎</span><h3>{running ? "Following vehicle tracks…" : "No vehicle results yet"}</h3><p>{running ? "Results arrive as the backend closes and joins each vehicle lifetime." : "Run a recorded video or RTSP source to see actual backend decisions."}</p></div>}</div>
        <footer className="results-footer"><span>One row per vehicle lifetime · No generated confidence scores</span><span role="status" className={cleanup === "Cleanup needs attention" ? "cleanup-warning" : ""}>{cleanup}</span></footer>
      </section><footer className="page-footer"><span>ANY VEHICLE <span className="footer-dot">/</span> VEHICLE INTELLIGENCE</span><span>Recorded video + RTSP · T1 backend</span></footer></>}
    </main>
  </div>;
}
export default App;
