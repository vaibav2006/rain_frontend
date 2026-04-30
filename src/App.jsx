import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

const BASE_URL = "https://ml-backend-4.onrender.com";
const api = (path) => {
  const normalizedPath = String(path || "").replace(/^\/api(?=\/)/, "");
  return `${BASE_URL}${normalizedPath}`;
};
const RESULT_PAGE_COUNT = 5;
const UPLOAD_STORAGE_KEY = "v2s_upload_details";
const AUTH_USERS_STORAGE_KEY = "v2s_auth_users";
const AUTH_SESSION_STORAGE_KEY = "v2s_auth_session";
const ADMIN_EMAIL = "admin@gmail.com";
const ADMIN_PASSWORD = "admin123";

const defaultFarmForm = {
  cropType: "Rice"
};

const cropOptions = ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane"];

function normalizeEmail(email) {
  return String(email || "").trim().toLowerCase();
}

function seedAdminUser(users) {
  const list = Array.isArray(users) ? [...users] : [];
  const hasAdmin = list.some((user) => normalizeEmail(user.email) === ADMIN_EMAIL);
  if (hasAdmin) {
    return list;
  }
  list.push({
    email: ADMIN_EMAIL,
    password: ADMIN_PASSWORD,
    role: "admin",
    createdAt: new Date().toISOString()
  });
  return list;
}

function readUsersFromStorage() {
  try {
    const raw = window.localStorage.getItem(AUTH_USERS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    const seeded = seedAdminUser(parsed);
    window.localStorage.setItem(AUTH_USERS_STORAGE_KEY, JSON.stringify(seeded));
    return seeded;
  } catch (error) {
    const seeded = seedAdminUser([]);
    window.localStorage.setItem(AUTH_USERS_STORAGE_KEY, JSON.stringify(seeded));
    return seeded;
  }
}

function saveUsersToStorage(users) {
  window.localStorage.setItem(AUTH_USERS_STORAGE_KEY, JSON.stringify(users));
}

function imageSource(base64) {
  return `data:image/png;base64,${base64}`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatMillimeterPerHour(value) {
  return `${Number(value || 0).toFixed(2)} mm/hr`;
}

function matrixToText(matrix) {
  return (matrix || [])
    .map((row) => row.map((value) => value.toFixed(3).padStart(7, " ")).join(" "))
    .join("\n");
}

function flattenVectorToText(vector) {
  return (vector || []).map((value) => value.toFixed(3)).join("  ");
}

async function readJsonSafe(response) {
  const raw = await response.text();
  if (!raw) {
    throw new Error(`Empty response body (HTTP ${response.status})`);
  }
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error(raw);
  }
}

function parseResultPage(pathname) {
  const match = (pathname || "").match(/^\/results(?:\/(\d+))?$/);
  if (!match) {
    return null;
  }
  if (!match[1]) {
    return 1;
  }
  const page = Number(match[1]);
  if (!Number.isInteger(page)) {
    return null;
  }
  return Math.max(1, Math.min(RESULT_PAGE_COUNT, page));
}

function formatFileSize(bytes) {
  const size = Number(bytes || 0);
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < (1024 * 1024)) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
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
  const location = useLocation();
  const navigate = useNavigate();
  const [authReady, setAuthReady] = useState(false);
  const [authUsers, setAuthUsers] = useState([]);
  const [session, setSession] = useState(null);
  const [authForm, setAuthForm] = useState({ email: "", password: "" });
  const [authError, setAuthError] = useState("");
  const [authNotice, setAuthNotice] = useState("");
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
  const [storedUploadDetails, setStoredUploadDetails] = useState(null);
  const resultSectionRef = useRef(null);
  const normalizedPath = location.pathname || "/";
  const isLoginRoute = normalizedPath === "/login";
  const isSignupRoute = normalizedPath === "/signup";
  const isAuthRoute = isLoginRoute || isSignupRoute;
  const isAdminRoute = normalizedPath === "/admin";
  const currentResultPage = parseResultPage(location.pathname);
  const isResultRoute = currentResultPage !== null;

  useEffect(() => {
    const users = readUsersFromStorage();
    setAuthUsers(users);

    try {
      const rawSession = window.localStorage.getItem(AUTH_SESSION_STORAGE_KEY);
      const parsedSession = rawSession ? JSON.parse(rawSession) : null;
      if (parsedSession?.email) {
        setSession(parsedSession);
      }
    } catch (error) {
      window.localStorage.removeItem(AUTH_SESSION_STORAGE_KEY);
    }

    setAuthReady(true);
  }, []);

  useEffect(() => {
    if (!authReady) {
      return;
    }

    if (!session && !isAuthRoute) {
      navigate("/login", { replace: true });
      return;
    }

    if (session && isAuthRoute) {
      navigate("/", { replace: true });
      return;
    }

    if (session && isAdminRoute && session.role !== "admin") {
      navigate("/", { replace: true });
    }
  }, [authReady, session, isAuthRoute, isAdminRoute, navigate]);

  useEffect(() => {
    let ignore = false;

    async function loadModelInfo() {
      if (!authReady || !session || isAuthRoute) {
        return;
      }
      try {
        const response = await fetch(api("/api/model-info"));
        const payload = await readJsonSafe(response);
        if (!response.ok) {
          throw new Error(payload.error || `HTTP ${response.status}`);
        }
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
        if (!ignore) {
          setError(fetchError.message || "Unable to reach the backend services.");
        }
      }
    }

    loadModelInfo();
    return () => {
      ignore = true;
    };
  }, [authReady, session, isAuthRoute]);

  useEffect(() => {
    if (!selectedFile) {
      return undefined;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(UPLOAD_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw);
      if (!parsed || !parsed.fileName) {
        return;
      }
      setStoredUploadDetails(parsed);
      if (parsed.previewDataUrl) {
        setPreviewUrl(parsed.previewDataUrl);
      }
      if (parsed.cropType) {
        setFarmForm({ cropType: parsed.cropType });
      }
    } catch (storageError) {
      console.warn("Unable to restore upload details from localStorage", storageError);
    }
  }, []);

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
    if (!predictionResult || !resultSectionRef.current || !isResultRoute) {
      return;
    }
    resultSectionRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [predictionResult, isResultRoute, currentResultPage]);

  useEffect(() => {
    if (!session || isAuthRoute) {
      return;
    }
    if (location.pathname === "/results") {
      navigate("/results/1", { replace: true });
    }
  }, [location.pathname, navigate, session, isAuthRoute]);

  useEffect(() => {
    if (!session || isAuthRoute) {
      return;
    }
    if (!isResultRoute) {
      return;
    }
    if (!predictionResult) {
      navigate("/", { replace: true });
    }
  }, [isResultRoute, predictionResult, navigate, session, isAuthRoute]);

  function persistUploadDetails(file, overrideCropType) {
    const activeCropType = overrideCropType || farmForm.cropType;
    if (!file) {
      setStoredUploadDetails(null);
      window.localStorage.removeItem(UPLOAD_STORAGE_KEY);
      return;
    }

    const basePayload = {
      fileName: file.name,
      fileSize: file.size,
      fileType: file.type || "unknown",
      lastModified: file.lastModified,
      cropType: activeCropType,
      storedAt: new Date().toISOString()
    };

    setStoredUploadDetails(basePayload);
    window.localStorage.setItem(UPLOAD_STORAGE_KEY, JSON.stringify(basePayload));

    const reader = new FileReader();
    reader.onload = () => {
      const enrichedPayload = {
        ...basePayload,
        previewDataUrl: typeof reader.result === "string" ? reader.result : ""
      };
      setStoredUploadDetails(enrichedPayload);
      window.localStorage.setItem(UPLOAD_STORAGE_KEY, JSON.stringify(enrichedPayload));
    };
    reader.readAsDataURL(file);
  }

  function goToResultPage(page) {
    const safePage = Math.max(1, Math.min(RESULT_PAGE_COUNT, Number(page) || 1));
    navigate(`/results/${safePage}`);
  }

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

  function handleAuthInputChange(event) {
    const { name, value } = event.target;
    setAuthForm((current) => ({ ...current, [name]: value }));
  }

  function handleSignup(event) {
    event.preventDefault();
    const email = normalizeEmail(authForm.email);
    const password = String(authForm.password || "").trim();

    if (!email || !password) {
      setAuthError("Email and password are required.");
      return;
    }
    if (password.length < 6) {
      setAuthError("Password must be at least 6 characters.");
      return;
    }
    if (email === ADMIN_EMAIL && password !== ADMIN_PASSWORD) {
      setAuthError("Admin account uses fixed credentials. Use admin123 for admin password.");
      return;
    }
    if (authUsers.some((user) => normalizeEmail(user.email) === email)) {
      setAuthError("This email is already registered. Please login.");
      return;
    }

    const role = email === ADMIN_EMAIL ? "admin" : "user";
    const newUser = {
      email,
      password,
      role,
      createdAt: new Date().toISOString()
    };
    const updatedUsers = [...authUsers, newUser];
    saveUsersToStorage(updatedUsers);
    setAuthUsers(updatedUsers);
    setAuthNotice("Signup successful. Please login with your new credentials.");
    setAuthError("");
    setAuthForm({ email, password: "" });
    navigate("/login", { replace: true });
  }

  function handleLogin(event) {
    event.preventDefault();
    const email = normalizeEmail(authForm.email);
    const password = String(authForm.password || "").trim();
    const matchedUser = authUsers.find(
      (user) => normalizeEmail(user.email) === email && String(user.password) === password
    );

    if (!matchedUser) {
      setAuthError("Invalid credentials. Please check your email and password.");
      return;
    }

    const nextSession = {
      email: matchedUser.email,
      role: matchedUser.role || "user",
      loggedInAt: new Date().toISOString()
    };
    window.localStorage.setItem(AUTH_SESSION_STORAGE_KEY, JSON.stringify(nextSession));
    setSession(nextSession);
    setAuthError("");
    setAuthNotice("");
    setAuthForm({ email: "", password: "" });
    navigate("/", { replace: true });
  }

  function handleLogout() {
    window.localStorage.removeItem(AUTH_SESSION_STORAGE_KEY);
    setSession(null);
    setPredictionResult(null);
    setSelectedFile(null);
    setPreviewUrl("");
    setAuthNotice("You have been logged out.");
    navigate("/login", { replace: true });
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

      const response = await fetch(api("/api/predict"), {
        method: "POST",
        body: formData
      });

      setBackgroundStatus("Prediction computed. Preparing explainable outputs and confidence chart...");
      const payload = await readJsonSafe(response);
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
      goToResultPage(1);
    } catch (predictError) {
      setError(predictError.message || "Prediction failed.");
      setBackgroundStatus("Prediction failed. Please check the error and try again.");
    } finally {
      setIsPredicting(false);
    }
  }

  const trainingHistory = predictionResult?.modelPerformance?.history || [];
  const lossSeries = (predictionResult?.modelPerformance?.lossSeries || []).map((item) => ({
    epoch: item.epoch,
    train: item.train,
    validation: item.validation
  }));
  const accuracySeries = (predictionResult?.modelPerformance?.accuracySeries || []).map((item) => ({
    epoch: item.epoch,
    train: item.train,
    validation: item.validation
  }));
  const datasetMetrics = predictionResult?.modelPerformance?.datasetMetrics || {};
  const activeResultPage = currentResultPage || 1;

  if (!authReady) {
    return (
      <div className="auth-shell">
        <div className="auth-card panel">
          <p className="section-label">Initializing</p>
          <h1>Preparing secure workspace...</h1>
          <p className="muted">Loading authentication and local session state.</p>
        </div>
      </div>
    );
  }

  if (!session || isAuthRoute) {
    const isSignupMode = isSignupRoute;
    return (
      <div className="auth-shell">
        <div className="background-glow background-glow-left" />
        <div className="background-glow background-glow-right" />
        <section className="auth-card panel">
          <p className="section-label">Secure Access</p>
          <h1>{isSignupMode ? "Create your account" : "Welcome back"}</h1>
          <p className="auth-subtitle">
            Login to continue with rainfall prediction workflows. Admin route is protected for{" "}
            <strong>{ADMIN_EMAIL}</strong>.
          </p>

          

          {authNotice ? <p className="auth-message auth-message-success">{authNotice}</p> : null}
          {authError ? <p className="auth-message auth-message-error">{authError}</p> : null}

          <form onSubmit={isSignupMode ? handleSignup : handleLogin} className="stacked-form auth-form">
            <label>
              <span>Email</span>
              <input
                type="email"
                name="email"
                value={authForm.email}
                onChange={handleAuthInputChange}
                placeholder="you@example.com"
                autoComplete="username"
                required
              />
            </label>
            <label>
              <span>Password</span>
              <input
                type="password"
                name="password"
                value={authForm.password}
                onChange={handleAuthInputChange}
                placeholder="Enter password"
                autoComplete={isSignupMode ? "new-password" : "current-password"}
                required
              />
            </label>

            <button type="submit" className="primary-button">
              {isSignupMode ? "Sign up" : "Login"}
            </button>
          </form>

          <div className="auth-switch">
            {isSignupMode ? (
              <p>
                Already registered? <Link to="/login">Go to login</Link>
              </p>
            ) : (
              <p>
                New user? <Link to="/signup">Create an account</Link>
              </p>
            )}
          </div>
        </section>
      </div>
    );
  }

  if (isAdminRoute && session.role === "admin") {
    return (
      <div className="page-shell">
        <div className="background-glow background-glow-left" />
        <div className="background-glow background-glow-right" />
        <main className="app-shell">
          <section className="panel admin-panel">
            <div className="admin-panel-header">
              <div>
                <p className="section-label">Admin Dashboard</p>
                <h1>Protected admin access</h1>
                <p className="muted">Only admin users can view registered credential records.</p>
              </div>
              <div className="admin-actions">
                <Link className="secondary-button route-button" to="/">Back to app</Link>
                <button type="button" className="secondary-button route-button" onClick={handleLogout}>Logout</button>
              </div>
            </div>
            <div className="admin-user-grid">
              {authUsers.map((user) => (
                <article key={`${user.email}-${user.createdAt}`} className="admin-user-card">
                  <p><strong>Email:</strong> {user.email}</p>
                  <p><strong>Role:</strong> {user.role || "user"}</p>
                  <p><strong>Created:</strong> {new Date(user.createdAt).toLocaleString()}</p>
                </article>
              ))}
            </div>
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="page-shell">
      <div className="background-glow background-glow-left" />
      <div className="background-glow background-glow-right" />
      {toast ? <PortalToast toast={toast} onClose={() => setToast(null)} /> : null}

      <main className="app-shell">
        <section className="panel app-toolbar">
          <div className="app-toolbar-left">
            <p className="section-label">Logged In</p>
            <h2>{session.email}</h2>
          </div>
          <div className="app-toolbar-actions">
            {session.role === "admin" ? (
              <Link className="secondary-button route-button" to="/admin">
                Admin panel
              </Link>
            ) : null}
            <button type="button" className="secondary-button route-button" onClick={handleLogout}>
              Logout
            </button>
          </div>
        </section>

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
                  onChange={(event) => {
                    const file = event.target.files?.[0] || null;
                    setSelectedFile(file);
                    persistUploadDetails(file);
                  }}
                />
              </label>

              <div className="field-grid farm-grid">
                <label>
                  <span>Crop type</span>
                  <select
                    value={farmForm.cropType}
                    onChange={(event) => {
                      const nextCropType = event.target.value;
                      setFarmForm((current) => ({ ...current, cropType: nextCropType }));
                      if (selectedFile) {
                        persistUploadDetails(selectedFile, nextCropType);
                      } else if (storedUploadDetails) {
                        const updatedDetails = {
                          ...storedUploadDetails,
                          cropType: nextCropType,
                          storedAt: new Date().toISOString()
                        };
                        setStoredUploadDetails(updatedDetails);
                        window.localStorage.setItem(UPLOAD_STORAGE_KEY, JSON.stringify(updatedDetails));
                      }
                    }}
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

              {storedUploadDetails ? (
                <div className="saved-upload-card">
                  <p className="section-label">Saved Upload Details</p>
                  <p><strong>Name:</strong> {storedUploadDetails.fileName}</p>
                  <p><strong>Type:</strong> {storedUploadDetails.fileType}</p>
                  <p><strong>Size:</strong> {formatFileSize(storedUploadDetails.fileSize)}</p>
                  <p><strong>Crop:</strong> {storedUploadDetails.cropType}</p>
                  <p className="muted">
                    Stored locally on this browser at{" "}
                    {storedUploadDetails.storedAt
                      ? new Date(storedUploadDetails.storedAt).toLocaleString()
                      : "unknown time"}
                    . Re-select the file to run a fresh prediction.
                  </p>
                </div>
              ) : null}

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
            <section className="panel result-route-panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Result Routing</p>
                  <h2>Prediction results are shown page by page</h2>
                </div>
                <div className="result-route-actions">
                  <button
                    type="button"
                    className="secondary-button route-button"
                    onClick={() => goToResultPage(activeResultPage - 1)}
                    disabled={!isResultRoute || activeResultPage <= 1}
                  >
                    Previous page
                  </button>
                  <button
                    type="button"
                    className="secondary-button route-button"
                    onClick={() => goToResultPage(activeResultPage + 1)}
                    disabled={!isResultRoute || activeResultPage >= RESULT_PAGE_COUNT}
                  >
                    Next page
                  </button>
                </div>
              </div>
              <div className="result-step-links">
                {[1, 2, 3, 4, 5].map((page) => (
                  <Link
                    key={`result-page-${page}`}
                    to={`/results/${page}`}
                    className={`result-step-link ${isResultRoute && activeResultPage === page ? "active" : ""}`}
                  >
                    Page {page}
                  </Link>
                ))}
              </div>
              {!isResultRoute ? (
                <p className="muted">Open a result page above to view section-wise outputs.</p>
              ) : null}
            </section>

            {isResultRoute && activeResultPage === 1 ? (
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
                  <p className="section-label">Rainfall And Error Metrics</p>
                  <h2>Rain intensity and dataset-level performance summary</h2>
                </div>
              </div>

              <div className="metrics-grid">
                <article className="metric-card">
                  <p className="metric-title">Estimated rainfall amount</p>
                  <h3>{formatMillimeterPerHour(predictionResult.rainfallAmount?.mmPerHour)}</h3>
                  <p className="muted">{predictionResult.rainfallAmount?.intensityBand}</p>
                  <p className="formula-text">{predictionResult.rainfallAmount?.formula}</p>
                </article>

                <article className="metric-card">
                  <p className="metric-title">Average MSE (whole dataset)</p>
                  <h3>{Number(datasetMetrics.averageMSE || 0).toFixed(6)}</h3>
                  <p className="muted">
                    Computed over {datasetMetrics.sampleCount || 0} samples after training.
                  </p>
                </article>

                <article className="metric-card">
                  <p className="metric-title">Average dataset accuracy</p>
                  <h3>{formatPercent(datasetMetrics.averageAccuracy || 0)}</h3>
                  <p className="muted">
                    Average loss: {Number(datasetMetrics.averageLoss || 0).toFixed(6)}
                  </p>
                </article>
              </div>
            </section>

              </>
            ) : null}

            {isResultRoute && activeResultPage === 2 ? (
              <>
            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Model Graphs</p>
                  <h2>Loss curves, accuracy visualization, and result bar charts</h2>
                </div>
              </div>

              {trainingHistory.length ? (
                <>
                  <div className="viz-grid">
                    <LineChartCard
                      title="Loss Graph"
                      subtitle="Train vs validation loss across epochs"
                      points={lossSeries}
                      series={[
                        { key: "train", label: "Train loss", color: "#67e8f9" },
                        { key: "validation", label: "Validation loss", color: "#fbbf24" }
                      ]}
                    />
                    <LineChartCard
                      title="Accuracy Graph"
                      subtitle="Train vs validation accuracy across epochs"
                      points={accuracySeries}
                      series={[
                        { key: "train", label: "Train accuracy", color: "#34d399" },
                        { key: "validation", label: "Validation accuracy", color: "#60a5fa" }
                      ]}
                    />
                    <LineChartCard
                      title="Validation MSE Graph"
                      subtitle="Validation MSE progression across epochs"
                      points={trainingHistory.map((item) => ({
                        epoch: item.epoch,
                        mse: item.validationMSE
                      }))}
                      series={[
                        { key: "mse", label: "Validation MSE", color: "#fb7185" }
                      ]}
                    />
                    <BarChartCard
                      title="Prediction Bar Chart"
                      subtitle="Current result distribution"
                      items={[
                        {
                          label: "No-rain probability",
                          value: predictionResult.prediction.probabilities?.[0]?.value || 0,
                          color: "#8a9aa7"
                        },
                        {
                          label: "Rain probability",
                          value: predictionResult.prediction.probabilities?.[1]?.value || 0,
                          color: "#29b6d1"
                        },
                        {
                          label: "Prediction confidence",
                          value: predictionResult.prediction.confidence || 0,
                          color: "#ffd08c"
                        },
                        {
                          label: "Dataset average MSE",
                          value: Math.min(1, Number(datasetMetrics.averageMSE || 0)),
                          color: "#f97316"
                        }
                      ]}
                    />
                  </div>

                </>
              ) : (
                <p className="muted">
                  Training history will appear after training metadata is available.
                </p>
              )}
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

              </>
            ) : null}

            {isResultRoute && activeResultPage === 3 ? (
              <>
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

              </>
            ) : null}

            {isResultRoute && activeResultPage === 4 ? (
              <>
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

              </>
            ) : null}

            {isResultRoute && activeResultPage === 5 ? (
              <>
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

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="section-label">Prediction Logs</p>
                  <h2>Backend processing logs after result generation</h2>
                </div>
              </div>
              <article className="log-card">
                <h3>Post-result prediction log</h3>
                {(predictionResult?.systemNotes?.predictionLog || predictionLogs || []).length ? (
                  <div className="log-list">
                    {(predictionResult?.systemNotes?.predictionLog || predictionLogs || []).map((line, index) => (
                      <p key={`post-result-log-${index}`}>{line}</p>
                    ))}
                  </div>
                ) : (
                  <p className="muted">No prediction logs available yet.</p>
                )}
              </article>
            </section>

              </>
            ) : null}

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

function LineChartCard({ title, subtitle, points, series }) {
  const width = 620;
  const height = 260;
  const padding = 28;
  const safePoints = points || [];

  const allValues = [];
  safePoints.forEach((point) => {
    series.forEach((line) => {
      const value = Number(point[line.key]);
      if (Number.isFinite(value)) {
        allValues.push(value);
      }
    });
  });

  const minValue = allValues.length ? Math.min(...allValues) : 0;
  const maxValue = allValues.length ? Math.max(...allValues) : 1;
  const valueRange = Math.max(maxValue - minValue, 1e-6);

  function xForIndex(index) {
    if (safePoints.length <= 1) {
      return width / 2;
    }
    return padding + ((width - (padding * 2)) * index) / (safePoints.length - 1);
  }

  function yForValue(value) {
    const normalized = (Number(value) - minValue) / valueRange;
    return (height - padding) - normalized * (height - (padding * 2));
  }

  return (
    <article className="viz-card">
      <h3>{title}</h3>
      <p className="muted">{subtitle}</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart" role="img" aria-label={title}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="chart-axis" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="chart-axis" />

        {series.map((line) => {
          const polylinePoints = safePoints
            .map((point, index) => `${xForIndex(index)},${yForValue(point[line.key])}`)
            .join(" ");

          return (
            <polyline
              key={line.key}
              points={polylinePoints}
              fill="none"
              stroke={line.color}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          );
        })}

        {safePoints.map((point, index) => (
          <text key={`epoch-${point.epoch}-${index}`} x={xForIndex(index)} y={height - 8} className="chart-label">
            {point.epoch}
          </text>
        ))}
      </svg>
      <div className="chart-legend">
        {series.map((line) => (
          <span key={line.key}>
            <i style={{ background: line.color }} />
            {line.label}
          </span>
        ))}
      </div>
    </article>
  );
}

function BarChartCard({ title, subtitle, items }) {
  const bars = items || [];
  return (
    <article className="viz-card">
      <h3>{title}</h3>
      <p className="muted">{subtitle}</p>
      <div className="chart-bars">
        {bars.map((item) => (
          <div key={item.label} className="chart-bar-row">
            <div className="probability-labels">
              <span>{item.label}</span>
              <strong>{formatPercent(item.value)}</strong>
            </div>
            <div className="probability-track">
              <div
                className="probability-fill"
                style={{
                  width: formatPercent(item.value),
                  background: `linear-gradient(90deg, ${item.color}, #ecfeff)`
                }}
              />
            </div>
          </div>
        ))}
      </div>
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