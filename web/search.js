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
const USER_NAME ="shrnik"
// ── Config ─────────────────────────────────────────────────────────────────
const HF_BASE_URL =
  `https://huggingface.co/datasets/${USER_NAME}/felidae-image-search/resolve/main`;

const IMAGE_BASE_URL =
  "https://storage.googleapis.com/public-datasets-lila/felidae-conservation-fund/";

const EMBEDDING_DIM = 512; // clip-vit-base-patch32
// ───────────────────────────────────────────────────────────────────────────

// ── State ──────────────────────────────────────────────────────────────────
let embeddings = null;   // Float32Array, shape [N, 512]
let metadata = [];       // [{id, file_name, category}, ...]
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

// ── Cached fetch — hits browser Cache API first, network on miss ───────────
const CACHE_NAME = "felidae-index-v1";

async function cachedFetch(url) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) return cached;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch failed: ${resp.status} ${url}`);
  await cache.put(url, resp.clone());
  return resp;
}

// ── Load precomputed data ──────────────────────────────────────────────────
async function loadIndex() {
  const metaUrl = `${HF_BASE_URL}/metadata.csv`;
  const embUrl  = `${HF_BASE_URL}/embeddings.bin`;

  setStatus("Loading metadata…", true);
  const csvResp = await cachedFetch(metaUrl);
  metadata = parseCSV(await csvResp.text());
  numImages = metadata.length;

  const embSizeMB = (numImages * EMBEDDING_DIM * 4 / 1e6).toFixed(0);
  setStatus(`Loading embeddings (${embSizeMB} MB)…`, true);
  const embResp = await cachedFetch(embUrl);
  const buffer = await embResp.arrayBuffer();
  embeddings = new Float32Array(buffer);

  if (embeddings.length !== numImages * EMBEDDING_DIM) {
    console.warn(
      `Embedding count mismatch: got ${embeddings.length} floats, ` +
      `expected ${numImages} × ${EMBEDDING_DIM}`
    );
  }

  // L2-normalise each image embedding in-place
  for (let i = 0; i < numImages; i++) {
    const offset = i * EMBEDDING_DIM;
    let norm = 0;
    for (let d = 0; d < EMBEDDING_DIM; d++) norm += embeddings[offset + d] ** 2;
    norm = Math.sqrt(norm);
    if (norm > 0) for (let d = 0; d < EMBEDDING_DIM; d++) embeddings[offset + d] /= norm;
  }
}

// ── CLIP text encoder via Transformers.js ──────────────────────────────────
let tokenizer = null;
let textModel = null;

async function encodeText(text) {
  if (!tokenizer || !textModel) {
    setStatus("Initialising CLIP encoder (first query only)…", true);
    const { AutoTokenizer, CLIPTextModelWithProjection } = await import(
      "https://cdn.jsdelivr.net/npm/@xenova/transformers"
    );
    [tokenizer, textModel] = await Promise.all([
      AutoTokenizer.from_pretrained("Xenova/clip-vit-base-patch32"),
      CLIPTextModelWithProjection.from_pretrained("Xenova/clip-vit-base-patch32"),
    ]);
  }

  const inputs = tokenizer([text], { padding: true, truncation: true });
  const { text_embeds } = await textModel(inputs);

  // L2 normalise
  const vec = text_embeds.data;
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  const result = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) result[i] = vec[i] / norm;
  return result;
}

// ── Cosine similarity search ───────────────────────────────────────────────
function topKSearch(queryVec, k, filterCategory) {
  const dim = EMBEDDING_DIM;
  const scores = [];

  const indices = Array.from({ length: numImages }, (_, i) => i);

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
