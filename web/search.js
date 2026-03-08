/**
 * Felidae Image Search — client-side CLIP semantic search
 *
 * How it works:
 *  1. Load precomputed float32 image embeddings from HuggingFace (embeddings.bin)
 *  2. Load metadata CSV (image file names + species labels)
 *  3. On search: encode the text query with the CLIP text encoder (ONNX)
 *  4. Dot-product similarity (vectors are already L2-normalised) → top-K results
 *
 * Configuration — update HF_BASE_URL to point at your HuggingFace dataset repo.
 */

// ── Config ─────────────────────────────────────────────────────────────────
const HF_BASE_URL =
  "https://huggingface.co/datasets/YOUR_USERNAME/felidae-image-search/resolve/main";

const IMAGE_BASE_URL =
  "https://lilawildlife.blob.core.windows.net/lila-wildlife/felidae-conservation-fund/";

const CLIP_ONNX_URL =
  "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_quantized.onnx";

const CLIP_TOKENIZER_URL =
  "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json";

const EMBEDDING_DIM = 512; // clip-vit-base-patch32
// ───────────────────────────────────────────────────────────────────────────

// ── State ──────────────────────────────────────────────────────────────────
let embeddings = null;   // Float32Array, shape [N, 512]
let metadata = [];       // [{id, file_name, category}, ...]
let onnxSession = null;
let tokenizer = null;
let numImages = 0;
// ───────────────────────────────────────────────────────────────────────────

// ── DOM refs ───────────────────────────────────────────────────────────────
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const queryEl = document.getElementById("query");
const searchBtn = document.getElementById("search-btn");
const categoryFilter = document.getElementById("category-filter");
const topKInput = document.getElementById("top-k");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxCaption = document.getElementById("lightbox-caption");
document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (e) => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLightbox(); });
// ───────────────────────────────────────────────────────────────────────────

function setStatus(msg, spin = false) {
  statusEl.innerHTML = spin
    ? `<span class="spinner"></span> ${msg}`
    : msg;
}

// ── CSV parser ─────────────────────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    return Object.fromEntries(headers.map((h, i) => [h, vals[i]]));
  });
}

// ── Load precomputed data ──────────────────────────────────────────────────
async function loadIndex() {
  setStatus("Loading metadata…", true);
  const csvResp = await fetch(`${HF_BASE_URL}/metadata.csv`);
  if (!csvResp.ok) throw new Error(`metadata.csv fetch failed: ${csvResp.status}`);
  metadata = parseCSV(await csvResp.text());
  numImages = metadata.length;

  // Populate category filter
  const cats = [...new Set(metadata.map((r) => r.category))].sort();
  cats.forEach((cat) => {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = cat.replace(/_/g, " ");
    categoryFilter.appendChild(opt);
  });

  setStatus(`Loading embeddings (${(numImages * EMBEDDING_DIM * 4 / 1e6).toFixed(0)} MB)…`, true);
  const embResp = await fetch(`${HF_BASE_URL}/embeddings.bin`);
  if (!embResp.ok) throw new Error(`embeddings.bin fetch failed: ${embResp.status}`);
  const buffer = await embResp.arrayBuffer();
  embeddings = new Float32Array(buffer);

  if (embeddings.length !== numImages * EMBEDDING_DIM) {
    console.warn(
      `Embedding count mismatch: got ${embeddings.length} floats, ` +
      `expected ${numImages} × ${EMBEDDING_DIM}`
    );
  }
}

// ── Load CLIP text encoder (ONNX) + tokenizer ──────────────────────────────
async function loadCLIP() {
  setStatus("Loading CLIP text encoder…", true);

  const [tokenizerResp] = await Promise.all([
    fetch(CLIP_TOKENIZER_URL),
  ]);
  if (!tokenizerResp.ok) throw new Error("Failed to fetch tokenizer");
  tokenizer = await tokenizerResp.json();

  ort.env.wasm.numThreads = 1;
  onnxSession = await ort.InferenceSession.create(CLIP_ONNX_URL, {
    executionProviders: ["wasm"],
  });
}

// ── Minimal BPE tokenizer for CLIP ────────────────────────────────────────
// We use the Xenova/transformers.js approach: fetch the tokenizer.json vocab
// and do BPE in JS. For simplicity we call the Transformers.js pipeline instead.

// Lazy-load @xenova/transformers for text tokenisation + feature extraction.
// This avoids bundling — works fine for a demo served from a local/CDN URL.
let clipTextPipeline = null;
async function getClipTextPipeline() {
  if (clipTextPipeline) return clipTextPipeline;

  setStatus("Initialising CLIP encoder (first query only)…", true);

  // Dynamically import Transformers.js from CDN
  const { pipeline } = await import(
    "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js"
  );

  clipTextPipeline = await pipeline(
    "feature-extraction",
    "Xenova/clip-vit-base-patch32",
    { quantized: true }
  );
  return clipTextPipeline;
}

// ── Encode a text query to a normalised 512-d float32 vector ──────────────
async function encodeText(text) {
  const pipe = await getClipTextPipeline();
  const output = await pipe(text, { pooling: "none" });
  // output.data is Float32Array of shape [1, seq_len, 512] — take [CLS] (or EOS)
  // For CLIP we want the EOS token embedding. Transformers.js returns the
  // last hidden state; the CLIP text embedding is the feature at the EOS position.
  // Use the last non-padding position: just take the final embedding.
  const dim = EMBEDDING_DIM;
  const seqLen = output.data.length / dim;
  // EOS is at seqLen-1
  const vec = new Float32Array(output.data.buffer, (seqLen - 1) * dim * 4, dim);

  // L2 normalise
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  const result = new Float32Array(dim);
  for (let i = 0; i < dim; i++) result[i] = vec[i] / norm;
  return result;
}

// ── Cosine similarity search ───────────────────────────────────────────────
function topKSearch(queryVec, k, filterCategory) {
  const dim = EMBEDDING_DIM;
  const scores = [];

  const indices = filterCategory
    ? metadata
        .map((r, i) => (r.category === filterCategory ? i : -1))
        .filter((i) => i >= 0)
    : Array.from({ length: numImages }, (_, i) => i);

  for (const idx of indices) {
    let dot = 0;
    const offset = idx * dim;
    for (let d = 0; d < dim; d++) dot += queryVec[d] * embeddings[offset + d];
    scores.push({ idx, score: dot });
  }

  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, k);
}

// ── Render results ─────────────────────────────────────────────────────────
function renderResults(topK) {
  resultsEl.innerHTML = "";

  if (topK.length === 0) {
    setStatus("No results.");
    return;
  }

  setStatus(`Showing top ${topK.length} results.`);

  for (const { idx, score } of topK) {
    const row = metadata[idx];
    const imgUrl = IMAGE_BASE_URL + row.file_name;
    const category = (row.category || "unknown").replace(/_/g, " ");

    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <img
        src="${imgUrl}"
        alt="${category}"
        loading="lazy"
        onerror="this.style.opacity='0.3'"
      />
      <div class="card-info">
        <div class="card-category">${category}</div>
        <div class="card-score">similarity ${score.toFixed(3)}</div>
      </div>`;

    card.addEventListener("click", () => openLightbox(imgUrl, category, row.file_name, score));
    resultsEl.appendChild(card);
  }
}

// ── Lightbox ───────────────────────────────────────────────────────────────
function openLightbox(url, category, fileName, score) {
  lightboxImg.src = url;
  lightboxImg.alt = category;
  lightboxCaption.textContent = `${category} · ${fileName} · similarity ${score.toFixed(3)}`;
  lightbox.classList.add("open");
}

function closeLightbox() {
  lightbox.classList.remove("open");
  lightboxImg.src = "";
}

// ── Search handler ─────────────────────────────────────────────────────────
async function doSearch() {
  const query = queryEl.value.trim();
  if (!query) return;

  searchBtn.disabled = true;
  setStatus("Encoding query…", true);

  try {
    const queryVec = await encodeText(query);
    const k = Math.max(1, Math.min(500, parseInt(topKInput.value, 10) || 48));
    const cat = categoryFilter.value;

    setStatus("Searching…", true);
    const results = topKSearch(queryVec, k, cat);
    renderResults(results);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    console.error(err);
  } finally {
    searchBtn.disabled = false;
  }
}

queryEl.addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
searchBtn.addEventListener("click", doSearch);

// ── Bootstrap ──────────────────────────────────────────────────────────────
(async () => {
  try {
    await loadIndex();
    setStatus(
      `Index loaded: ${numImages.toLocaleString()} images. ` +
      `Type a query and press Search.`
    );
    searchBtn.disabled = false;
    queryEl.focus();
  } catch (err) {
    setStatus(
      `Failed to load index: ${err.message}. ` +
      `Check that HF_BASE_URL in search.js points to your HuggingFace repo ` +
      `and that embeddings.bin / metadata.csv have been uploaded.`
    );
    console.error(err);
  }
})();
