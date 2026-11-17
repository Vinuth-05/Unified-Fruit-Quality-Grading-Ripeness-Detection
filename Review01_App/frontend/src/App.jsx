import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";
const COLORS = { Ripe: "#2cc985", Unripe: "#f5bd3f", Overripe: "#ff5964", Review: "#ff9f43", Unknown: "#9aa4b2" };
const n = (v) => Number.isFinite(Number(v)) ? Number(v) : 0;
const pct = (v) => `${n(v).toFixed(1)}%`;

function Stat({ label, value, hint, tone = "pink" }) {
  return <div className={`stat stat-${tone}`}><span>{label}</span><strong>{value}</strong>{hint && <small>{hint}</small>}</div>;
}
function Count({ label, value, color }) {
  return <div className="count"><i style={{ background: color }} /><span>{label}</span><strong>{n(value)}</strong></div>;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const prediction = result?.prediction;
  const detections = Array.isArray(prediction?.detections) ? prediction.detections : [];
  const fruit = prediction?.fruit_counts || {};
  const ripe = prediction?.ripeness_counts || {};
  const grades = prediction?.grade_counts || {};
  const reviewed = detections.filter((d) => d.is_unknown || d.fruit_type_mismatch).length;
  const avgDet = useMemo(() => detections.length ? detections.reduce((s, d) => s + n(d.detection_confidence), 0) / detections.length : 0, [detections]);
  const accepted = detections.filter((d) => !d.is_unknown && !d.fruit_type_mismatch);
  const avgVit = useMemo(() => accepted.length ? accepted.reduce((s, d) => s + n(d.confidence), 0) / accepted.length : 0, [accepted]);
  const annotatedUrl = result?.annotated_image_url ? `${API_BASE}${result.annotated_image_url}` : "";

  function chooseImage(e) {
    const next = e.target.files?.[0];
    if (!next) return;
    if (!/\.(jpe?g|png|webp)$/i.test(next.name)) { setError("Choose a JPG, PNG, or WEBP image."); return; }
    if (preview) URL.revokeObjectURL(preview);
    setFile(next); setPreview(URL.createObjectURL(next)); setResult(null); setError("");
  }

  async function analyze() {
    if (!file) { setError("Choose an image first."); return; }
    setBusy(true); setError(""); setResult(null);
    const body = new FormData(); body.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/predict-v3`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.detail || "Analysis failed. Check that the backend is running.");
      setResult(data);
    } catch (e) { setError(e.message || "Could not connect to the backend at 127.0.0.1:8000."); }
    finally { setBusy(false); }
  }

  function reset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(""); setResult(null); setError("");
  }

  return <div className="fv-app">
    <header className="fv-header"><div className="fv-mark">🍎</div><div><div className="eyebrow">YOLO DETECTION · AI RIPENESS ANALYSIS</div><h1>FruitVision <em>AI</em></h1><p>Detect each fruit. Grade its ripeness. Flag uncertain results.</p></div></header>
    <main className="fv-main">
      {!result && <section className="upload-card">
        <div className="upload-copy"><span className="eyebrow">IMAGE ANALYSIS</span><h2>One image. Clear results.</h2><p>Upload a photo containing apples, oranges, or pears. For best results, use a sharp image with fruits clearly visible and not heavily overlapping.</p>
          <label className="file-button">{file ? "CHANGE IMAGE" : "CHOOSE IMAGE"}<input type="file" accept=".jpg,.jpeg,.png,.webp" onChange={chooseImage} /></label>
          {file && <><div className="file-name">{file.name}</div><button className="primary-button" onClick={analyze} disabled={busy}>{busy ? "ANALYZING…" : "ANALYZE IMAGE"}</button></>}
          {busy && <p className="busy-note">YOLO detection → ResNet18 fruit check → ViT ripeness classification</p>}
        </div>
        <div className="preview-box">{preview ? <img src={preview} alt="Selected fruit image" /> : <div className="preview-placeholder"><span>＋</span><strong>Your image preview</strong><small>JPG · PNG · WEBP</small></div>}</div>
      </section>}
      {error && <div className="error-box" role="alert">{error}</div>}
      {result && prediction && <>
        <section className="report-head"><div><span className="eyebrow">ANALYSIS REPORT</span><h2>Fruit quality summary</h2><p>{prediction.image_name || result.filename} · {n(prediction.total_detected_fruits)} detections</p></div><button className="secondary-button" onClick={reset}>← New image</button></section>
        <section className="summary-grid">
          <Stat label="FRUITS DETECTED" value={n(prediction.total_detected_fruits)} hint="YOLO bounding boxes" />
          <Stat label="ACCEPTED GRADES" value={n(prediction.accepted_classifications)} hint="Passed consistency checks" tone="green" />
          <Stat label="NEEDS REVIEW" value={reviewed} hint="Check these manually" tone="orange" />
          <Stat label="AVG. DETECTION SCORE" value={pct(avgDet)} hint="YOLO localization confidence" tone="blue" />
        </section>
        <section className="image-card"><div className="section-title"><div><span className="eyebrow">VISUAL RESULTS</span><h3>Detected fruit map</h3></div>{annotatedUrl && <a href={annotatedUrl} target="_blank" rel="noreferrer">Open image ↗</a>}</div>
          {annotatedUrl ? <img className="annotated" src={annotatedUrl} alt="Annotated fruit detections" /> : <p>No annotated image was returned by the API.</p>}
          <div className="legend"><span><i style={{background:COLORS.Ripe}}/>Ripe</span><span><i style={{background:COLORS.Unripe}}/>Unripe</span><span><i style={{background:COLORS.Overripe}}/>Overripe</span><span><i style={{background:COLORS.Review}}/>Needs review</span><span><i style={{background:COLORS.Unknown}}/>Unknown</span></div>
        </section>
        <section className="counts-section"><div className="section-title"><div><span className="eyebrow">AT A GLANCE</span><h3>Counts by fruit, ripeness and grade</h3></div></div>
          <div className="count-group"><h4>Fruit type</h4><div className="count-grid"><Count label="Apple" value={fruit.Apple} color="#e55b73"/><Count label="Orange" value={fruit.Orange} color="#f39b32"/><Count label="Pear" value={fruit.Pear} color="#82bd61"/></div></div>
          <div className="count-group"><h4>Ripeness <small>accepted classifications only</small></h4><div className="count-grid"><Count label="Ripe" value={ripe.Ripe} color={COLORS.Ripe}/><Count label="Unripe" value={ripe.Unripe} color={COLORS.Unripe}/><Count label="Overripe" value={ripe.Overripe} color={COLORS.Overripe}/></div></div>
          <div className="count-group"><h4>Project quality grades</h4><div className="count-grid"><Count label="Grade A · Ripe" value={grades["Grade A"]} color={COLORS.Ripe}/><Count label="Grade B · Unripe" value={grades["Grade B"]} color={COLORS.Unripe}/><Count label="Grade C · Overripe" value={grades["Grade C"]} color={COLORS.Overripe}/></div></div>
        </section>
        <section className="details-section"><div className="section-title"><div><span className="eyebrow">PER-FRUIT BREAKDOWN</span><h3>Detection details</h3></div><div className="small-stat">Average accepted ViT score <strong>{pct(avgVit)}</strong></div></div>
          {!detections.length ? <div className="empty-box"><strong>No supported fruit detected.</strong><p>Try a clearer image with larger, less-overlapped fruit.</p></div> : <div className="detail-grid">{detections.map((d) => {
            const review = Boolean(d.is_unknown || d.fruit_type_mismatch);
            const state = review ? "Needs review" : (d.ripeness || "Unknown");
            const color = review ? COLORS.Review : (COLORS[d.ripeness] || COLORS.Unknown);
            return <article className="detail-card" key={d.detection_id} style={{"--state-color":color}}>
              <div className="detail-top"><span className="detail-number">{d.detection_id}</span><div><strong>{d.detected_fruit_type || d.fruit_type || "Unknown fruit"}</strong><small>{review ? (d.rejection_reason || d.consistency_warning || "Manual check recommended") : `${d.ripeness} · ${d.quality_grade}`}</small></div><span className="status">{review ? "REVIEW" : state.toUpperCase()}</span></div>
              <div className="score-row"><div><small>YOLO detection</small><strong>{pct(d.detection_confidence)}</strong></div>{d.verifier_confidence !== undefined && <div><small>ResNet18 fruit check</small><strong>{pct(d.verifier_confidence)}</strong></div>}<div><small>ViT ripeness</small><strong>{pct(d.confidence)}</strong></div></div>
              <div className="grade-row"><span>Quality grade</span><strong>{review ? "Needs Review" : (d.quality_grade || "Unknown")}</strong></div>
              {d.consistency_warning && <p className="warning">{d.consistency_warning}</p>}
              {!review && d.recommendation && <p className="recommendation">{d.recommendation}</p>}
            </article>;
          })}</div>}
        </section>
        <p className="disclaimer">Grade mapping is project-defined: Ripe → A, Unripe → B, Overripe → C. Scores are model confidence values, not guarantees. Fruit marked “Needs Review” is excluded from ripeness and grade totals.</p>
      </>}
    </main>
    <footer className="fv-footer">FRUITVISION AI <span>·</span> Unified Fruit Quality Grading & Ripeness Detection</footer>
  </div>;
}
