import { useMemo, useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

const LOCALIZATION_OPTIONS = [
  "scalp",
  "ear",
  "face",
  "back",
  "trunk",
  "chest",
  "upper extremity",
  "abdomen",
  "unknown",
  "lower extremity",
  "genital",
  "neck",
  "hand",
  "foot",
  "acral",
];

const CANCER_CLASSES = [
  "Actinic Keratoses (akiec)",
  "Basal Cell Carcinoma (bcc)",
  "Melanoma (mel)",
];

// Returns a readable error message from an Axios or JavaScript error.
function getErrorMessage(error) {
  const detail = error.response?.data?.detail;

  if (detail) {
    // ── Skin validator rejection ──
    if (typeof detail === "object" && detail.error === "Image rejected by skin validator.") {
      return `Invalid image — ${detail.reason} (Skin confidence: ${(detail.skin_confidence * 100).toFixed(1)}%)`;
    }

    // ── Plain string error ──
    if (typeof detail === "string") {
      return detail;
    }

    // ── Any other object error ──
    if (typeof detail === "object" && detail.reason) {
      return detail.reason;
    }

    return JSON.stringify(detail);
  }



  if (error.code === "ERR_NETWORK") {
    return "Cannot connect to the API. Start FastAPI on http://127.0.0.1:8000 first.";
  }

  return error.message || "Something went wrong while making the prediction.";
}

// Builds the multipart form body required by the FastAPI /predict endpoint.
function createPredictionFormData({ image, age, sex, localization }) {
  const formData = new FormData();

  formData.append("image", image);
  formData.append("age", age);
  formData.append("sex", sex);
  formData.append("localization", localization);

  return formData;
}

// Converts the API class name into a simple cancerous/non-cancerous category.
function getLesionCategory(prediction) {
  return CANCER_CLASSES.includes(prediction)
    ? "Cancerous Lesion"
    : "Non-Cancerous Lesion";
}

// Renders one row inside the top predictions list.
function TopPredictionItem({ item, index }) {
  return (
    <li className="top-prediction-item">
      <span className="rank">{index + 1}</span>
      <span className="class-name">{item.class_name}</span>
      <strong>{item.confidence}%</strong>
    </li>
  );
}

// Renders the model result after the API returns a prediction.
function PredictionResult({ result }) {
  const category = result.category || getLesionCategory(result.prediction);
  const isLowConfidence =
    result.is_low_confidence === true || Number(result.confidence) < 55;

      if (isLowConfidence) {
        return (
          <section className="result-panel">
            <div className="notice warning">
              No clear skin lesion detected. Please upload a proper dermoscopic skin lesion image.
            </div>
        
            <div className="summary-grid">
              <div>
                <span>Confidence</span>
                <strong>{result.confidence}%</strong>
              </div>
            </div>
          </section>
        );
      }
    

  return (
    <section className="result-panel" aria-live="polite">
      <div className="result-header">
        <div>
          <p className="eyebrow">Prediction</p>
          <h2>{result.prediction}</h2>
        </div>
        <div className={isLowConfidence ? "score warning" : "score"}>
          {result.confidence}%
        </div>
      </div>

      <div className={isLowConfidence ? "notice warning" : "notice success"}>
        {isLowConfidence
          ? "No clear skin lesion detected. Please upload a proper dermoscopic skin lesion image."
          : "Prediction completed successfully."}
      </div>

      <div className="summary-grid">
        <div>
          <span>Category</span>
          <strong>{category}</strong>
        </div>
        <div>
          <span>Confidence</span>
          <strong>{result.confidence}%</strong>
        </div>
      </div>

      {result.top_predictions?.length > 0 && (
        <div className="top-predictions">
          <h3>Top Predictions</h3>
          <ol>
            {result.top_predictions.map((item, index) => (
              <TopPredictionItem
                key={item.class_name}
                item={item}
                index={index}
              />
            ))}
          </ol>
        </div>
      ) }

      {result.llm_explanation && (
  <div className="llm-explanation">
    <h3>AI Explanation</h3>
    <p>{result.llm_explanation}</p>
  </div>)}
    </section>
  );
}



// Renders the uploaded image beside the Grad-CAM heatmap returned by FastAPI.
function GradCamViewer({ originalImageUrl, gradcamImageUrl }) {
  if (!originalImageUrl || !gradcamImageUrl) {
    return null;
  }

  return (
    <section className="gradcam-panel">
      <div className="gradcam-header">
        <p className="eyebrow">Model attention</p>
        <h2>Grad-CAM Visualization</h2>
      </div>

      <div className="gradcam-grid">
        <figure>
          <img src={originalImageUrl} alt="Original uploaded lesion" />
          <figcaption>Original Image</figcaption>
        </figure>

        <figure>
          <img src={gradcamImageUrl} alt="Grad-CAM heatmap overlay" />
          <figcaption>Grad-CAM Heatmap</figcaption>
        </figure>
      </div>
    </section>
  );
}

// Main React application that replaces the Streamlit upload and prediction UI.
function App() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [age, setAge] = useState(45);
  const [sex, setSex] = useState("male");
  const [localization, setLocalization] = useState("back");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isRejected, setIsRejected] = useState(false); 

  const canPredict = useMemo(() => image && age !== "", [image, age]);

  // Stores the selected image file and creates a local browser preview.
  function handleImageChange(event) {
    const selectedFile = event.target.files?.[0];
    setResult(null);
    setError("");
    setIsRejected(false);  
    setImage(selectedFile || null);
    setPreviewUrl(selectedFile ? URL.createObjectURL(selectedFile) : "");
  }

  // Sends the image and metadata to FastAPI and stores the prediction response.
  async function handlePredict(event) {
    event.preventDefault();

    if (!image) {
      setError("Please upload a skin lesion image before predicting.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);
    setIsRejected(false); 

    try {
      const formData = createPredictionFormData({
        image,
        age,
        sex,
        localization,
      });

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30000,
      });

      setResult(response.data);
    } catch (requestError) {
      // ── Check if it is a validator rejection specifically ──
      const detail = requestError.response?.data?.detail;
      if (
        requestError.response?.status === 400 &&
        typeof detail === "object" &&
        detail.error === "Image rejected by skin validator."
      ) {
        setIsRejected(true);   // ← flag it separately
      }
      setError(getErrorMessage(requestError));
    } finally {
      setIsLoading(false);
    }
  }

  // Clears the current form, image preview, errors, and prediction result.
  function handleReset() {
    setImage(null);
    setPreviewUrl("");
    setAge(45);
    setSex("male");
    setLocalization("back");
    setResult(null);
    setError("");
    setIsRejected(false);
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="header">
          <div>
            <p className="eyebrow">Multimodal classifier</p>
            <h1>Skin Lesion Detection</h1>
          </div>
          <span className="api-pill">API: {API_URL}</span>
        </div>

        <form className="prediction-layout" onSubmit={handlePredict}>
          <section className="upload-panel">
            <label className="file-drop">
              <input
                accept="image/png,image/jpeg,image/jpg"
                type="file"
                onChange={handleImageChange}
              />
              {previewUrl ? (
                <div className="preview-wrapper">
                  <img src={previewUrl} alt="Selected lesion preview" />
                  {isRejected && (
                    <div className="rejected-badge">❌ Not a skin image</div>
                  )}
                </div>
              ) : (
                <span>Choose dermoscopic image</span>
              )}
            </label>

          </section>

          <section className="form-panel">
            <label>
              <span>Age</span>
              <input
                max="120"
                min="0"
                type="number"
                value={age}
                onChange={(event) => setAge(event.target.value)}
              />
            </label>

            <label>
              <span>Sex</span>
              <select
                value={sex}
                onChange={(event) => setSex(event.target.value)}
              >
                <option value="male">male</option>
                <option value="female">female</option>
              </select>
            </label>

            <label>
              <span>Localization</span>
              <select
                value={localization}
                onChange={(event) => setLocalization(event.target.value)}
              >
                {LOCALIZATION_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>

            <div className="actions">
              <button disabled={!canPredict || isLoading} type="submit">
                {isLoading ? "Predicting..." : "Predict"}
              </button>
              <button className="secondary" type="button" onClick={handleReset}>
                Reset
              </button>
            </div>

            {error && <div className="notice error">{error}</div>}
            {isRejected && (
              <div className="notice error">
                Image rejected by skin validator.
              </div>
            )}
          </section>
        </form>

        {result && <PredictionResult result={result} />}

        {result?.gradcam_image &&
         !result?.is_low_confidence && (
          <GradCamViewer
            originalImageUrl={previewUrl}
            gradcamImageUrl={result.gradcam_image}
          />
        )}

        <p className="disclaimer">
          Predictions are for research purposes only and should not be
          considered medical diagnosis.
        </p>
      </section>
    </main>
  );
}

export default App;

