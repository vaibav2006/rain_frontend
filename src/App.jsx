import { useEffect, useRef, useState } from "react";
const BASE_URL = "https://ml-backend-4.onrender.com";
const defaultFarmForm = {
  cropType: "Rice"
};

const cropOptions = ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane"];

function imageSource(base64) {
  return `data:image/png;base64,${base64}`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function matrixToText(matrix) {
  return (matrix || [])
    .map((row) => row.map((value) => value.toFixed(3).padStart(7, " ")).join(" "))
    .join("\n");
}

function flattenVectorToText(vector) {
  return (vector || []).map((value) => value.toFixed(3)).join("  ");
}

function buildInitialBackendReadiness(payload) {
  const notes = ["Frontend connected to the gateway and ML service endpoints."];
  notes.push(
    payload?.modelReady
      ? "Model artifacts are available and ready for prediction."
      : "Model artifacts are not loaded yet. They will be initialized automatically on the first prediction."
  );

  const infoNotes = payload?.logs || [];
  return notes.concat(infoNotes);
}

function buildPredictionBackendReadiness(payload) {
  const notes = [
    "Prediction request completed successfully through the backend pipeline.",
    "Uploaded image preprocessing and CNN inference finished without backend interruption."
  ];
  const modelLogs = payload?.modelLifecycleLogs || [];
  const loadedWeightsNote = modelLogs.find((line) => line.includes("Loaded the saved CNN weights"));
  if (loadedWeightsNote) {
    notes.push(loadedWeightsNote);
  }
  return notes;
}

function extractTrainingLogFromPrediction(payload) {
  const lifecycleLogs = payload?.modelLifecycleLogs || [];
  const trainingLines = lifecycleLogs.filter((line) => {
    const normalized = line.toLowerCase();
    return (
      normalized.includes("training")
      || normalized.includes("epoch")
      || normalized.includes("dataset")
    );
  });

  if (trainingLines.length) {
    return trainingLines;
  }

  if (lifecycleLogs.some((line) => line.includes("Loaded the saved CNN weights"))) {
    return ["No retraining was needed. The backend reused the saved CNN weights."];
  }

  return ["Training lifecycle did not change during this prediction request."];
}

function logPredictionMathematicsToConsole(payload) {
  const mathTrace = payload?.mathematics?.layerMathTrace || [];
  const consoleLines = payload?.mathematics?.consoleLines || [];

  console.groupCollapsed("Prediction logs");
  (payload?.logs || []).forEach((line) => console.log(line));
  console.groupEnd();

  console.groupCollapsed("Layer mathematics");
  mathTrace.forEach((item) => {
    console.groupCollapsed(item.name);
    console.log("Formula:", item.formula);
    console.log("Equation:", item.equation);
    (item.details || []).forEach((detail) => console.log(detail));
    if (item.patchPreview) {
      console.log("Patch preview");
      console.table(item.patchPreview);
    }
    if (item.kernelPreview) {
      console.log("Kernel preview");
      console.table(item.kernelPreview);
    }
    if (item.windowPreview) {
      console.log("Pooling window preview");
      console.table(item.windowPreview);
    }
    if (item.topContributions?.length) {
      console.log("Top dense contributions");
      console.table(item.topContributions);
    }
    console.groupEnd();
  });
  console.groupEnd();

  console.groupCollapsed("Mathematical summary lines");
  consoleLines.forEach((line) => console.log(line));
  console.groupEnd();

  if (payload?.xai) {
    console.groupCollapsed("Grad-CAM explainability");
    console.log(payload.xai.title);
    console.log(payload.xai.highlight);
    console.log(payload.xai.summary);
    console.table(payload.xai.topFilters || []);
    console.table(payload.xai.focusRegions || []);
    console.groupEnd();
  }

  if (payload?.farmerAlert) {
    console.groupCollapsed("Smart Farm Rain Alert");
    console.log("Crop:", payload.farmerAlert.cropType);
    console.log("Advisory:", payload.farmerAlert.advisory);
    console.log("Risk band:", payload.farmerAlert.riskBand);
    console.log("Toast alert eligible:", payload.farmerAlert.shouldNotify);
    console.table(payload.farmerAlert.confidenceChart?.points || []);
    console.groupEnd();
  }

  console.log("Prediction payload", payload);
}

function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [modelReady, setModelReady] = useState(false);
  const [healthLogs, setHealthLogs] = useState([]);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [predictionLogs, setPredictionLogs] = useState([]);
  const [farmForm, setFarmForm] = useState(defaultFarmForm);
  const [predictionResult, setPredictionResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [backgroundStatus, setBackgroundStatus] = useState(
    "Ready. Upload an image and click predict."
  );
  const [error, setError] = useState("");
  const [notificationStatus, setNotificationStatus] = useState(
    "Toast alert status will appear here after prediction."
  );
  const [toast, setToast] = useState(null);
  const [matrixZoom, setMatrixZoom] = useState(1);
  const resultSectionRef = useRef(null);

  useEffect(() => {
    let ignore = false;

    async function loadModelInfo() {
  try {
    const response = await fetch(`${BASE_URL}/api/model-info`);

    const text = await response.text();
    console.log("RAW MODEL INFO:", text);

    const payload = JSON.parse(text);

    if (!ignore) {
      setModelInfo(payload.modelInfo);
      setModelReady(Boolean(payload.modelReady));
      setHealthLogs(buildInitialBackendReadiness(payload));
      setTrainingLogs(
        payload.modelReady
          ? ["Model is ready. New training lifecycle updates will appear after prediction requests."]
          : ["Model is not trained yet. The first prediction will initialize training automatically."]
      );
      setPredictionLogs([]);
    }
  } catch (fetchError) {
    console.log("MODEL INFO ERROR:", fetchError);

    if (!ignore) {
      setError(fetchError.message || "Unable to reach the backend services.");
    }
  }
}

    loadModelInfo();
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  useEffect(() => {
    if (!toast) {
      return undefined;
    }
    const timer = window.setTimeout(() => {
      setToast(null);
    }, 4200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!predictionResult || !resultSectionRef.current) {
      return;
    }
    resultSectionRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [predictionResult]);

  function triggerToastAlert(alertPayload) {
    if (!alertPayload?.shouldNotify) {
      setToast({
        title: "Rain Alert Not Triggered",
        message: "Rain confidence stayed below 70%, so the advisory remains informational only.",
        variant: "info"
      });
      setNotificationStatus(
        "Rain confidence is below the 70% threshold, so the in-app toast alert was not triggered."
      );
      return;
    }
    setToast({
      title: alertPayload.notificationTitle,
      message: alertPayload.notificationBody,
      variant: "success"
    });
    setNotificationStatus("Rain alert toast displayed successfully inside the portal.");
  }

  async function handlePredict(event) {
    event.preventDefault();
    if (!selectedFile) {
      setError("Choose an image before running prediction.");
      return;
    }

    setError("");
    setIsPredicting(true);
    setBackgroundStatus("Background processing started: preprocessing image and running CNN inference.");
    setNotificationStatus("Checking whether a rain alert toast should be shown.");

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      formData.append("cropType", farmForm.cropType);

      const response = await fetch(`${BASE_URL}/predict`, {
        method: "POST",
        body: formData
      });

      setBackgroundStatus("Prediction computed. Preparing explainable outputs and confidence chart...");
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Prediction failed.");
      }

      logPredictionMathematicsToConsole(payload);
      triggerToastAlert(payload.farmerAlert);

      setPredictionResult(payload);
      setModelInfo(payload.modelInfo);
      setModelReady(true);
      setHealthLogs(
        payload?.systemNotes?.backendReadiness || buildPredictionBackendReadiness(payload)
      );
      setTrainingLogs(payload?.systemNotes?.trainingLog || extractTrainingLogFromPrediction(payload));
      setPredictionLogs(payload?.systemNotes?.predictionLog || payload?.logs || []);
      setBackgroundStatus("Completed. Results are now visible below.");
    } catch (predictError) {
      setError(predictError.message || "Prediction failed.");
      setBackgroundStatus("Prediction failed. Please check the error and try again.");
    } finally {
      setIsPredicting(false);
    }
  }

  return (
    <div className="page-shell">
      <div className="background-glow background-glow-left" />
      <div className="background-glow background-glow-right" />
      {toast ? <PortalToast toast={toast} onClose={() => setToast(null)} /> : null}

      <main className="app-shell">
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow portal-name">V2S Rainfall Prediction And Crop Safety Portal</p>
            <h1 className="hero-title">
              Human-centered rainfall intelligence for explainable prediction and crop safety.
            </h1>
            <p className="hero-text">
              A presentation-ready portal that transforms uploaded imagery into explainable rainfall
              insight with CNN mathematics, Grad-CAM attention mapping, toast-based risk alerts,
              and SDG-linked crop safety guidance.
            </p>
            <div className="hero-chips">
              <span className="chip">
                {modelInfo ? `Input ${modelInfo.architecture.inputSize}x${modelInfo.architecture.inputSize}` : "Model pending"}
              </span>
              <span className="chip">First free XAI rainfall tool</span>
              <span className="chip">SDG 2 - Zero Hunger</span>
              <span className="chip">SDG 8 - Economic Growth</span>
              <span className="chip">SDG 13 - Climate Action</span>
              <span className="chip">
                <a
                  href={modelInfo?.paperAlignment?.referenceUrl || "/v2s-ieee-paper.html"}
                  target="_blank"
                  rel="noreferrer"
                >
                  Paper reference
                </a>
              </span>
            </div>
          </div>

          <div className="hero-card panel">
            <p className="section-label">Novelty Highlights</p>
            {modelInfo ? (
              <>
                <div className="feature-promo-card">
                  <h3>Why did AI predict rain? - Explainable AI</h3>
                  <p className="muted">
                    Grad-CAM highlights the cloud regions that pushed the model toward the rain class.
                  </p>
                  <div className="badge-row">
                    <span className="badge-pill innovation-pill">First free XAI rainfall tool</span>
                    <span className="badge-pill">SDG 13 - Climate Action</span>
                  </div>
                </div>
                <div className="feature-promo-card">
                  <h3>Smart Farm Rain Alert</h3>
                  <p className="muted">
                    Toast-style alerts and crop-specific advisory appear when cloud-image rain confidence exceeds 70%.
                  </p>
                  <div className="badge-row">
                    <span className="badge-pill">SDG 2</span>
                    <span className="badge-pill">SDG 8</span>
                    <span className="badge-pill">SDG 13</span>
                  </div>
                </div>
                <div className="stat-row">
                  <span>Status</span>
                  <strong>{modelReady ? "Model ready" : "Waiting for first prediction"}</strong>
                </div>
                <div className="stat-row">
                  <span>Pooling</span>
                  <strong>
                    {modelInfo.architecture.poolSize}x{modelInfo.architecture.poolSize} stride {modelInfo.architecture.poolStride}
                  </strong>
                </div>
                <div className="stat-row">
                  <span>Hidden units</span>
                  <strong>{modelInfo.architecture.hiddenUnits}</strong>
                </div>
                <div className="stat-row">
                  <span>Classes</span>
                  <strong>{(modelInfo.classLabels || []).join(" / ")}</strong>
                </div>
              </>
            ) : (
              <p className="muted">Connecting to the backend services.</p>
            )}
          </div>
        </section>

        <section className="workspace-grid workspace-grid-single">
          <div className="panel action-panel">
            <p className="section-label">Image Upload</p>
            <form onSubmit={handlePredict} className="stacked-form">
              <label className="upload-box">
                <span className="upload-title">Select any image for prediction</span>
                <span className="upload-hint">Any image format supported by your browser</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
                />
              </label>

              <div className="field-grid farm-grid">
                <label>
                  <span>Crop type</span>
                  <select
                    value={farmForm.cropType}
                    onChange={(event) =>
                      setFarmForm((current) => ({ ...current, cropType: event.target.value }))
                    }
                  >
                    {cropOptions.map((crop) => (
                      <option key={crop} value={crop}>
                        {crop}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <p className="helper-text">
                Rainfall alerting and advisory are now generated purely from cloud-image confidence
                and selected crop type.
              </p>

              {previewUrl ? (
                <div className="preview-frame">
                  <img src={previewUrl} alt="Selected preview" />
                </div>
              ) : (
                <div className="empty-preview">The selected image preview will appear here.</div>
              )}

              <button type="submit" className="primary-button" disabled={isPredicting}>
                {isPredicting ? "Running prediction..." : "Predict rainfall from image"}
              </button>
              <div className="background-status-banner" role="status" aria-live="polite">
                {backgroundStatus}
              </div>
            </form>
          </div>
        </section>

        {error ? (
          <section className="panel error-panel">
            <p className="section-label">Error</p>
            <p>{error}</p>
          </section>
        ) : null}

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="section-label">System Notes</p>
              <h2>What the app is doing</h2>
            </div>
          </div>

          <div className="log-grid">
            <LogCard title="Backend readiness" logs={healthLogs} />
            <LogCard title="Training log" logs={trainingLogs} />
            <LogCard title="Prediction log" logs={predictionLogs} />
          </div>
        </section>

        {predictionResult ? (
          <>
            <section className="panel" ref={resultSectionRef}>
              <div className="section-heading">
                <div>
                  <p className="section-label">Prediction Summary</p>
                  <h2>{predictionResult.prediction.label === "rain" ? "Rain likely" : "No rain likely"}</h2>
                </div>
                <div className="confidence-pill">
                  Confidence {formatPercent(predictionResult.prediction.confidence)}
                </div>
              </div>

              <div className="summary-grid">
                <div className="summary-image">
                  <img
                    src={imageSource(predictionResult.uploadedPreviewBase64)}
                    alt="Preprocessed upload preview"
                  />
                </div>
                <div className="summary-details">
                  {(predictionResult.prediction.probabilities || []).map((item) => (
                    <div key={item.label} className="probability-row">
                      <div className="probability-labels">
                        <span>{item.label}</span>
                        <strong>{formatPercent(item.value)}</strong>
                      </div>
                      <div className="probability-track">
                        <div className="probability-fill" style={{ width: formatPercent(item.value) }} />
                      </div>
                    </div>
                  ))}
                  <div className="badge-row">
                    <span className="badge-pill innovation-pill">
                      {predictionResult.xai.innovationBadge}
                    </span>
                    <span className="badge-pill">{predictionResult.xai.sdgBadge}</span>
                    {(predictionResult.farmerAlert.sdgBadges || []).map((badge) => (
                      <span key={badge} className="badge-pill">
                        {badge}
                      </span>
                    ))}
                  </div>
                  <div className="math-box">
                    <p className="math-label">Model input tensor preview</p>
                    <pre>{matrixToText(predictionResult.mathematics.inputTensorPreview)}</pre>
                    <p className="math-label">Full 28 x 28 input matrix used by the CNN</p>
                    <div className="matrix-slider-row">
                      <label htmlFor="matrix-zoom-slider">
                        Matrix zoom: {matrixZoom.toFixed(2)}x
                      </label>
                      <input
                        id="matrix-zoom-slider"
                        type="range"
                        min="0.65"
                        max="1.9"
                        step="0.05"
                        value={matrixZoom}
                        onChange={(event) => setMatrixZoom(Number(event.target.value))}
                      />
                    </div>
                    <p className="muted">
                      Use the slider and scrollbars to inspect all 28 x 28 matrix values clearly.
                    </p>
                    <div className="matrix-scroll-frame">
                      <pre style={{ fontSize: `${0.74 * matrixZoom}rem` }}>
                        {matrixToText(predictionResult.mathematics.inputTensor28x28)}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Confidence Analytics</p>
                  <h2>Rainfall confidence chart from the uploaded cloud image</h2>
                </div>
              </div>

              <div className="confidence-chart-grid">
                <article className="confidence-chart-card">
                  <div
                    className="confidence-gauge"
                    style={{
                      "--gauge-value": `${Math.max(
                        0,
                        Math.min(
                          100,
                          (predictionResult.confidenceChart?.points?.find((point) => point.label === "Rain Probability")?.value
                            ?? predictionResult.prediction.probabilities?.[1]?.value
                            ?? 0) * 100
                        )
                      )}%`
                    }}
                  >
                    <div className="confidence-gauge-core">
                      <p>Rain</p>
                      <strong>
                        {formatPercent(
                          predictionResult.confidenceChart?.points?.find((point) => point.label === "Rain Probability")?.value
                          ?? predictionResult.prediction.probabilities?.[1]?.value
                          ?? 0
                        )}
                      </strong>
                    </div>
                  </div>
                  <p className="muted">
                    Decision threshold: {formatPercent(predictionResult.confidenceChart?.threshold || 0.7)}
                  </p>
                  <p className="muted">{predictionResult.farmerAlert?.confidenceNarrative}</p>
                </article>

                <article className="confidence-chart-card">
                  {(predictionResult.confidenceChart?.points || []).map((point) => (
                    <div key={point.label} className="confidence-bar-row">
                      <div className="probability-labels">
                        <span>{point.label}</span>
                        <strong>{formatPercent(point.value)}</strong>
                      </div>
                      <div className="probability-track">
                        <div
                          className="probability-fill"
                          style={{
                            width: formatPercent(point.value),
                            background: `linear-gradient(90deg, ${point.color}, #f0f6ff)`
                          }}
                        />
                      </div>
                      <p className="muted">{point.description}</p>
                    </div>
                  ))}
                </article>
              </div>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">CNN Ordered Flow</p>
                  <h2>Layer-by-layer pipeline and matrix transformations in exact order</h2>
                </div>
              </div>
              <ol className="ordered-layer-list">
                <li>
                  <strong>Input tensor (1 x 28 x 28)</strong>
                  <p>
                    Single-channel preprocessed cloud image. This full 28 x 28 matrix is shown in the matrix viewer.
                  </p>
                </li>
                <li>
                  <strong>Convolution layer (4 filters, kernel 3 x 3, same-padding)</strong>
                  <p>{predictionResult.layerGroups?.[0]?.formula}</p>
                  <p>{predictionResult.layerGroups?.[0]?.description}</p>
                  <p>Output: 4 x 28 x 28</p>
                </li>
                <li>
                  <strong>ReLU activation layer</strong>
                  <p>{predictionResult.layerGroups?.[1]?.formula}</p>
                  <p>{predictionResult.layerGroups?.[1]?.description}</p>
                  <p>Output: 4 x 28 x 28</p>
                </li>
                <li>
                  <strong>Max-pooling layer (window 4 x 4, stride 2)</strong>
                  <p>{predictionResult.layerGroups?.[2]?.formula}</p>
                  <p>{predictionResult.layerGroups?.[2]?.description}</p>
                  <p>
                    Output: 4 x {predictionResult.layerGroups?.[2]?.maps?.[0]?.stats?.shape?.join(" x ") || "13 x 13"}
                  </p>
                </li>
                <li>
                  <strong>Flatten layer</strong>
                  <p>4 x 13 x 13 = 676</p>
                  <p>Output: vector length 676</p>
                  <p>
                    Full flatten output is displayed in the CNN Layer Views section as a 1 x 676 array and image.
                  </p>
                </li>
                <li>
                  <strong>Dense hidden layer (676 -&gt; 24) + ReLU</strong>
                  <p>{predictionResult.denseGroups?.[0]?.formula}</p>
                  <p>{predictionResult.denseGroups?.[0]?.description}</p>
                </li>
                <li>
                  <strong>Output dense logits (24 -&gt; 2) then softmax</strong>
                  <p>{predictionResult.denseGroups?.[1]?.formula}</p>
                  <p>
                    Final output order: [P(no-rain), P(rain)] = [
                    {formatPercent(predictionResult.prediction.probabilities?.[0]?.value || 0)},{" "}
                    {formatPercent(predictionResult.prediction.probabilities?.[1]?.value || 0)}]
                  </p>
                </li>
                <li>
                  <strong>Explainability layer (Grad-CAM)</strong>
                  <p>{predictionResult.xai?.formula}</p>
                  <p>{predictionResult.xai?.summary}</p>
                </li>
              </ol>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Novelty 1</p>
                  <h2>{predictionResult.xai.title}</h2>
                </div>
              </div>

              <div className="xai-grid">
                <article className="xai-card">
                  <h3>Grad-CAM Heatmap</h3>
                  <img src={imageSource(predictionResult.xai.heatmapBase64)} alt="Grad-CAM heatmap" />
                  <div className="legend-row">
                    {(predictionResult.xai.legend || []).map((item) => (
                      <div key={item.label} className="legend-item">
                        <span className="legend-swatch" style={{ background: item.color }} />
                        <span>{item.label}</span>
                      </div>
                    ))}
                  </div>
                </article>

                <article className="xai-card">
                  <h3>Attention Overlay</h3>
                  <img src={imageSource(predictionResult.xai.overlayBase64)} alt="Heatmap overlay" />
                  <div className="mini-stats">
                    <span>mean {predictionResult.xai.heatmapStats.mean}</span>
                    <span>max {predictionResult.xai.heatmapStats.max}</span>
                    <span>std {predictionResult.xai.heatmapStats.std}</span>
                  </div>
                </article>

                <article className="xai-card xai-copy-card">
                  <h3>{predictionResult.xai.highlight}</h3>
                  <p>{predictionResult.xai.summary}</p>
                  <p className="formula-text">{predictionResult.xai.formula}</p>
                  <div className="badge-row">
                    <span className="badge-pill innovation-pill">
                      {predictionResult.xai.innovationBadge}
                    </span>
                    <span className="badge-pill">{predictionResult.xai.sdgBadge}</span>
                  </div>
                  <div className="info-grid">
                    <div>
                      <h4>Top filter weights</h4>
                      {(predictionResult.xai.topFilters || []).map((item) => (
                        <p key={item.filter}>
                          Filter {item.filter}: {item.weight.toFixed(4)}
                        </p>
                      ))}
                    </div>
                    <div>
                      <h4>Focus regions</h4>
                      {(predictionResult.xai.focusRegions || []).slice(0, 4).map((item) => (
                        <p key={item.name}>
                          {item.name}: {item.score.toFixed(4)}
                        </p>
                      ))}
                    </div>
                  </div>
                </article>
              </div>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Novelty 2</p>
                  <h2>{predictionResult.farmerAlert.title}</h2>
                </div>
              </div>

              <div className="alert-grid">
                <article className="alert-card">
                  <h3>Crop-specific advisory</h3>
                  <div className="status-row">
                    <span>Crop type</span>
                    <strong>{predictionResult.farmerAlert.cropType}</strong>
                  </div>
                  <div className="status-row">
                    <span>Risk band</span>
                    <strong>{predictionResult.farmerAlert.riskBand}</strong>
                  </div>
                  <div className="status-row">
                    <span>Alert threshold</span>
                    <strong>{formatPercent(predictionResult.farmerAlert.alertThreshold)}</strong>
                  </div>
                  <p className="advisory-copy">{predictionResult.farmerAlert.advisory}</p>
                  <div className="notification-banner">
                    <strong>Confidence narrative</strong>
                    <span>{predictionResult.farmerAlert.confidenceNarrative}</span>
                  </div>
                  <div className="trace-preview-block">
                    <p className="trace-label">Weekly crop checklist</p>
                    <div className="log-list">
                      {(predictionResult.farmerAlert.weeklyChecklist || []).map((item) => (
                        <p key={item}>{item}</p>
                      ))}
                    </div>
                  </div>
                  <div className="notification-banner">
                    <strong>Toast alert</strong>
                    <span>{notificationStatus}</span>
                  </div>
                  <div className="badge-row">
                    {(predictionResult.farmerAlert.sdgBadges || []).map((badge) => (
                      <span key={badge} className="badge-pill">
                        {badge}
                      </span>
                    ))}
                  </div>
                </article>
              </div>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Layer Mathematics</p>
                  <h2>Mathematically computed results for every key stage</h2>
                </div>
              </div>

              <div className="math-trace-grid">
                {(predictionResult.mathematics.layerMathTrace || []).map((item) => (
                  <MathTraceCard key={item.name} item={item} />
                ))}
              </div>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Preprocessing Pipeline</p>
                  <h2>From uploaded image to CNN-ready tensor</h2>
                </div>
              </div>

              <div className="stage-grid">
                {(predictionResult.preprocessingStages || []).map((stage) => (
                  <article key={stage.name} className="stage-card">
                    <img src={imageSource(stage.imageBase64)} alt={stage.name} />
                    <div className="stage-copy">
                      <h3>{stage.name}</h3>
                      <p>{stage.description}</p>
                      <p className="formula-text">{stage.formula}</p>
                      <div className="mini-stats">
                        <span>shape {stage.stats.shape.join(" x ")}</span>
                        <span>mean {stage.stats.mean}</span>
                        <span>std {stage.stats.std}</span>
                      </div>
                      <pre>{matrixToText(stage.matrixPreview)}</pre>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Mentor Matrix Trace</p>
                  <h2>6x6 to 4x4 to 3x3 split matrices for each filter</h2>
                </div>
              </div>
              <p className="muted">
                Method: strongest-activation 6x6 crop from each ReLU filter map, then stride-1 max pooling
                (3x3 window gives 4x4, followed by 2x2 window gives 3x3).
              </p>
              <div className="matrix-flow-grid">
                {(predictionResult.mathematics.matrixProgression || []).map((trace) => (
                  <article key={`matrix-trace-${trace.filterIndex}`} className="matrix-flow-card">
                    <h3>Filter {trace.filterIndex}</h3>
                    <p className="muted">{trace.method}</p>
                    <p className="formula-text">{trace.formulas.sixBySix}</p>
                    <img src={imageSource(trace.sixBySix.imageBase64)} alt={`Filter ${trace.filterIndex} 6x6`} />
                    <pre>{matrixToText(trace.sixBySix.matrix)}</pre>
                    <p className="formula-text">{trace.formulas.fourByFour}</p>
                    <img src={imageSource(trace.fourByFour.imageBase64)} alt={`Filter ${trace.filterIndex} 4x4`} />
                    <pre>{matrixToText(trace.fourByFour.matrix)}</pre>
                    <p className="formula-text">{trace.formulas.threeByThree}</p>
                    <img src={imageSource(trace.threeByThree.imageBase64)} alt={`Filter ${trace.filterIndex} 3x3`} />
                    <pre>{matrixToText(trace.threeByThree.matrix)}</pre>
                  </article>
                ))}
              </div>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">CNN Layer Views</p>
                  <h2>Feature maps after each neural network stage</h2>
                </div>
              </div>

              <div className="layer-group-stack">
                {(predictionResult.layerGroups || []).map((group) => (
                  <section key={group.name} className="layer-group">
                    <div className="layer-header">
                      <div>
                        <h3>{group.name}</h3>
                        <p>{group.description}</p>
                      </div>
                      <p className="formula-text">{group.formula}</p>
                    </div>

                    <div className="map-grid">
                      {(group.maps || []).map((map) => (
                        <article key={map.name} className="map-card">
                          <img
                            src={imageSource(map.imageBase64)}
                            alt={map.name}
                            className={group.name === "Flatten Layer" ? "flatten-strip-image" : ""}
                          />
                          {group.name === "Flatten Layer" ? (
                            <p className="muted">Flatten strip visualization (1 x 676)</p>
                          ) : null}
                          <h4>{map.name}</h4>
                          <div className="mini-stats">
                            <span>shape {map.stats.shape.join(" x ")}</span>
                            <span>min {map.stats.min}</span>
                            <span>max {map.stats.max}</span>
                            <span>mean {map.stats.mean}</span>
                          </div>
                          {map.fullMatrix ? (
                            <div className="full-matrix-frame">
                              <pre>
                                {group.name === "Flatten Layer"
                                  ? flattenVectorToText(map.fullVector || [])
                                  : matrixToText(map.fullMatrix)}
                              </pre>
                            </div>
                          ) : (
                            <pre>{matrixToText(map.matrixPreview)}</pre>
                          )}
                        </article>
                      ))}
                    </div>
                  </section>
                ))}
              </div>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Dense Mathematics</p>
                  <h2>Final vectors and class reasoning</h2>
                </div>
              </div>

              <div className="dense-grid">
                {(predictionResult.denseGroups || []).map((group) => (
                  <article key={group.name} className="dense-card">
                    <h3>{group.name}</h3>
                    <p>{group.description}</p>
                    <p className="formula-text">{group.formula}</p>
                    <div className="bar-stack">
                      {(group.topActivations || []).map((item) => (
                        <div key={`${group.name}-${item.index}`} className="bar-row">
                          <div className="bar-label">
                            <span>Index {item.index}</span>
                            <strong>{item.value.toFixed(4)}</strong>
                          </div>
                          <div className="probability-track">
                            <div
                              className="probability-fill dense-fill"
                              style={{
                                width: `${Math.max(6, Math.min(100, Math.abs(item.value) * 100))}%`
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                    <pre>{matrixToText(group.matrixPreview)}</pre>
                  </article>
                ))}
              </div>
            </section>

          </>
        ) : null}
      </main>

    </div>
  );
}

function LogCard({ title, logs }) {
  return (
    <article className="log-card">
      <h3>{title}</h3>
      {logs.length ? (
        <div className="log-list">
          {logs.map((line, index) => (
            <p key={`${title}-${index}`}>{line}</p>
          ))}
        </div>
      ) : (
        <p className="muted">No messages yet.</p>
      )}
    </article>
  );
}

function MathTraceCard({ item }) {
  return (
    <article className="math-trace-card">
      <h3>{item.name}</h3>
      <p className="formula-text">{item.formula}</p>
      <p className="equation-text">{item.equation}</p>
      {(item.details || []).map((detail) => (
        <p key={detail} className="muted">
          {detail}
        </p>
      ))}

      {item.patchPreview ? (
        <div className="trace-preview-block">
          <p className="trace-label">Patch preview</p>
          <pre>{matrixToText(item.patchPreview)}</pre>
        </div>
      ) : null}

      {item.kernelPreview ? (
        <div className="trace-preview-block">
          <p className="trace-label">Kernel preview</p>
          <pre>{matrixToText(item.kernelPreview)}</pre>
        </div>
      ) : null}

      {item.windowPreview ? (
        <div className="trace-preview-block">
          <p className="trace-label">Pooling window</p>
          <pre>{matrixToText(item.windowPreview)}</pre>
        </div>
      ) : null}

      {item.topContributions?.length ? (
        <div className="trace-preview-block">
          <p className="trace-label">Top contributions</p>
          <div className="contribution-list">
            {item.topContributions.map((contribution) => (
              <div key={contribution.index} className="contribution-item">
                <span>Index {contribution.index}</span>
                <span>weight {contribution.weight}</span>
                <span>activation {contribution.activation}</span>
                <span>contribution {contribution.contribution}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </article>
  );
}

function PortalToast({ toast, onClose }) {
  return (
    <div className={`portal-toast ${toast.variant || "info"}`}>
      <div className="toast-accent" />
      <div className="toast-body">
        <p className="toast-label">Live Alert</p>
        <h3>{toast.title}</h3>
        <p>{toast.message}</p>
      </div>
      <button type="button" className="toast-close" onClick={onClose} aria-label="Close alert">
        x
      </button>
    </div>
  );
}

export default App;



