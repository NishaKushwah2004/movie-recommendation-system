# 🎬 Movie Recommendation System

A full-stack movie discovery and recommendation application combining local machine-learning models with live data from The Movie Database (TMDB). Users can browse trending films, search by keyword, view detailed movie pages, and receive personalized recommendations powered by TF-IDF content similarity and genre-based filtering.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Tech Stack](#tech-stack)
- [API Reference](#api-reference)
- [Data & ML Models](#data--ml-models)
- [Frontend (Streamlit)](#frontend-streamlit)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Known Limitations & Future Work](#known-limitations--future-work)

---

## Overview

This project is split into two layers:

| Layer | Technology | Responsibility |
|---|---|---|
| **Backend** | FastAPI (Python) | ML inference, TMDB proxy, recommendation logic |
| **Frontend** | Streamlit (Python) | UI, search, poster grid, detail pages |

The backend exposes a REST API that wraps both a locally-trained TF-IDF similarity model and the TMDB REST API. The Streamlit frontend communicates exclusively with the FastAPI backend — it never calls TMDB directly.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Browser / User                     │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────┐
│            Streamlit Frontend  (app.py)             │
│  • Home feed (trending / popular / top_rated …)     │
│  • Keyword search with dropdown autocomplete        │
│  • Movie detail page                                │
│  • Recommendation grids (TF-IDF + Genre)            │
└───────────────────────┬─────────────────────────────┘
                        │ REST (JSON)
┌───────────────────────▼─────────────────────────────┐
│             FastAPI Backend  (main.py)              │
│                                                     │
│  ┌─────────────────┐   ┌──────────────────────────┐ │
│  │  TF-IDF Engine  │   │     TMDB API Proxy       │ │
│  │  (local .pkl)   │   │  (httpx async client)    │ │
│  │                 │   │                          │ │
│  │  df.pkl         │   │  /trending               │ │
│  │  indices.pkl    │   │  /search/movie           │ │
│  │  tfidf.pkl      │   │  /movie/{id}             │ │
│  │  tfidf_matrix   │   │  /discover/movie         │ │
│  └────────┬────────┘   └────────────┬─────────────┘ │
│           │                         │               │
│           └────────────┬────────────┘               │
│                        │                            │
│              Bundle Response                        │
│         (details + tfidf + genre)                   │
└─────────────────────────────────────────────────────┘
                        │ HTTPS
┌───────────────────────▼─────────────────────────────┐
│               api.themoviedb.org                    │
└─────────────────────────────────────────────────────┘
```

### Request flow — Movie Detail Page

1. User selects a movie (from search or home feed); Streamlit navigates to `?view=details&id={tmdb_id}`.
2. Streamlit calls `GET /movie/id/{tmdb_id}` → returns poster, backdrop, genres, overview.
3. Streamlit calls `GET /movie/search?query={title}` (bundle endpoint):
   - FastAPI resolves the title against its local TF-IDF matrix → top-N similar movie titles.
   - For each similar title, FastAPI calls TMDB `/search/movie` to fetch a poster URL.
   - FastAPI also runs a TMDB `/discover/movie` query using the primary genre.
   - Returns `tfidf_recommendations` + `genre_recommendations` in a single response.
4. Streamlit renders two separate poster grids.

---

## Directory Structure

```
movie-recommendation-system/
│
├── main.py               # FastAPI application (backend)
├── app.py                # Streamlit application (frontend)
├── requirements.txt      # Python dependencies
│
├── df.pkl                # Pickled DataFrame of movie metadata
├── indices.pkl           # Title → DataFrame-index mapping
├── tfidf.pkl             # Fitted TfidfVectorizer object
└── tfidf_matrix.pkl      # Sparse TF-IDF feature matrix
```

> The four `.pkl` files are generated offline during model training. They must be present at runtime; the FastAPI startup event (`@app.on_event("startup")`) loads all four and raises a `RuntimeError` if any are missing or malformed.

---

## Tech Stack

### Backend

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.111.0 | Async REST API framework |
| `uvicorn` | 0.30.1 | ASGI server |
| `httpx` | 0.27.0 | Async HTTP client for TMDB calls |
| `pydantic` | (bundled) | Request/response schema validation |
| `numpy` | 2.1.0 | Cosine similarity computation |
| `scipy` | 1.13.1 | Sparse matrix operations |
| `pandas` | 2.2.2 | Movie metadata DataFrame |
| `scikit-learn` | 1.5.1 | TF-IDF vectorization |
| `python-dotenv` | 1.0.1 | Environment variable loading |

### Frontend

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.36.0 | Interactive web UI |
| `requests` | (stdlib) | HTTP calls to FastAPI backend |

### External APIs

| Service | Usage |
|---|---|
| [TMDB API v3](https://developer.themoviedb.org/docs) | Poster images, movie metadata, search, discover |

---

## API Reference

All endpoints are served by the FastAPI backend. Base URL: `http://localhost:8000` (local) or the deployed Render URL.

---

### `GET /health`

Simple liveness check.

**Response**
```json
{ "status": "ok" }
```

---

### `GET /home`

Returns a list of movie cards for the Streamlit home feed.

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `category` | string | `popular` | One of: `trending`, `popular`, `top_rated`, `now_playing`, `upcoming` |
| `limit` | integer | `24` | Number of results (1–100) |

**Response** — `List[TMDBMovieCard]`
```json
[
  {
    "tmdb_id": 550,
    "title": "Fight Club",
    "release_date": "1999-10-15",
    "poster_url": "https://image.tmdb.org/t/p/w500/...",
    "vote_average": 8.4
  }
]
```

---

### `GET /tmdb/search`

Proxies a TMDB keyword search. Returns the raw TMDB response shape (with a `results` array) so the Streamlit client can parse it directly.

**Query parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✅ | Search keyword |
| `page` | integer | — | Page number (default: 1) |

**Response** — raw TMDB `{ results: [...] }` object.

---

### `GET /movie/id/{tmdb_id}`

Fetches full movie details from TMDB for a given ID.

**Path parameters**

| Parameter | Type | Description |
|---|---|---|
| `tmdb_id` | integer | TMDB movie ID |

**Response** — `TMDBMovieDetails`
```json
{
  "tmdb_id": 550,
  "title": "Fight Club",
  "overview": "...",
  "release_date": "1999-10-15",
  "poster_url": "https://image.tmdb.org/t/p/w500/...",
  "backdrop_url": "https://image.tmdb.org/t/p/w500/...",
  "genres": [{ "id": 18, "name": "Drama" }]
}
```

---

### `GET /recommend/genre`

Returns genre-based movie recommendations via TMDB Discover.

**Query parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `tmdb_id` | integer | ✅ | Source movie ID |
| `limit` | integer | — | Number of results (default: 18, max: 50) |

**Response** — `List[TMDBMovieCard]` (excludes the source movie itself)

---

### `GET /recommend/tfidf`

Returns TF-IDF content-based recommendations using the local dataset.

**Query parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `title` | string | ✅ | Movie title (must match a title in the local dataset) |
| `top_n` | integer | — | Number of results (default: 10, max: 50) |

**Response**
```json
[
  { "title": "Se7en", "score": 0.842 },
  { "title": "The Usual Suspects", "score": 0.791 }
]
```

---

### `GET /movie/search` ⭐ Bundle Endpoint

The primary endpoint used by the detail page. Returns movie details, TF-IDF recommendations (with TMDB posters), and genre recommendations in a single response.

**Query parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✅ | Movie title to search for |
| `tfidf_top_n` | integer | — | TF-IDF results to return (default: 12) |
| `genre_limit` | integer | — | Genre discover results (default: 12) |

**Response** — `SearchBundleResponse`
```json
{
  "query": "Fight Club",
  "movie_details": { ... },
  "tfidf_recommendations": [
    {
      "title": "Se7en",
      "score": 0.842,
      "tmdb": { "tmdb_id": 807, "poster_url": "...", ... }
    }
  ],
  "genre_recommendations": [ ... ]
}
```

---

## Data & ML Models

### Pickle files

| File | Contents | Notes |
|---|---|---|
| `df.pkl` | `pandas.DataFrame` with at minimum a `title` column | Used to look up titles and retrieve metadata |
| `indices.pkl` | `dict` or `pandas.Series` mapping `title → DataFrame index` | Keys are lowercased and stripped; duplicates resolve to the last index |
| `tfidf.pkl` | Fitted `sklearn.feature_extraction.text.TfidfVectorizer` | Stored but not used at inference time (matrix is pre-computed) |
| `tfidf_matrix.pkl` | Sparse CSR matrix (`scipy.sparse.csr_matrix`) of shape `(n_movies, n_features)` | Cosine similarity computed as `matrix @ query_vector.T` |

### Similarity computation

```python
# Get the TF-IDF row for the query movie
qv = tfidf_matrix[idx]

# Dot product with every other row = cosine similarity (vectors already L2-normalised)
scores = (tfidf_matrix @ qv.T).toarray().ravel()

# Sort descending, exclude self
order = np.argsort(-scores)
```

The vectorizer should be trained with `norm='l2'` so that the dot product equals cosine similarity. If your vectorizer used a different norm, scores are proportional but not strictly cosine similarity.

---

## Frontend (Streamlit)

### Navigation model

Streamlit is a single-file app (`app.py`) that implements client-side routing via `st.session_state` and `st.query_params`:

| State | URL | Description |
|---|---|---|
| `view = "home"` | `?view=home` | Home feed + search |
| `view = "details"` | `?view=details&id={tmdb_id}` | Movie detail + recommendation page |

Both `goto_home()` and `goto_details(tmdb_id)` update both `session_state` and `query_params` then call `st.rerun()`.

### Key components

**`poster_grid(cards, cols, key_prefix)`**
Renders a responsive grid of movie posters with "Open" buttons. Each button calls `goto_details(tmdb_id)`.

**`parse_tmdb_search_to_cards(data, keyword, limit)`**
Normalises two API response shapes into a unified list of `{tmdb_id, title, poster_url}` dicts, then filters by keyword containment. Falls back to the full result list if no keyword matches are found, preventing blank search results.

**`to_cards_from_tfidf_items(tfidf_items)`**
Extracts the nested `tmdb` card from each TF-IDF recommendation item returned by `/movie/search`.

### Caching

`@st.cache_data(ttl=30)` is applied to all API calls via `api_get_json`. The 30-second TTL balances autocomplete freshness against redundant network requests.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A TMDB API key (free at [themoviedb.org](https://www.themoviedb.org/settings/api))
- The four trained `.pkl` files in the project root

### Local development

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd movie-recommendation-system

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file
echo "TMDB_API_KEY=your_key_here" > .env

# 5. Start the FastAPI backend
uvicorn main:app --reload --port 8000

# 6. In a separate terminal, start the Streamlit frontend
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TMDB_API_KEY` | ✅ | Your TMDB v3 API key. The backend raises `ValueError` on startup if this is absent. |

---

## Deployment

The application is deployed on [Render](https://render.com).

- **Backend**: `https://movie-recommendation-system-o8da.onrender.com`
- The Streamlit `app.py` hardcodes `API_BASE` to the Render URL as its primary target.

### Render configuration (backend)

```yaml
# render.yaml (example)
services:
  - type: web
    name: movie-rec-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: TMDB_API_KEY
        sync: false
```

> **Note**: Render free-tier instances spin down after inactivity. The first request after a cold start may take 30–60 seconds while the instance wakes and re-loads the pickle files into memory.

---

## Pydantic Schemas

```
TMDBMovieCard
  tmdb_id       int
  title         str
  release_date  str?
  poster_url    str?
  vote_average  float?

TMDBMovieDetails (extends TMDBMovieCard)
  overview      str?
  backdrop_url  str?
  genres        List[{id: int, name: str}]

TFIDFRecItem
  title         str
  score         float
  tmdb          TMDBMovieCard?

SearchBundleResponse
  query                  str
  movie_details          TMDBMovieDetails
  tfidf_recommendations  List[TFIDFRecItem]
  genre_recommendations  List[TMDBMovieCard]
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| TMDB network timeout | `httpx.RequestError` → HTTP 502 with message |
| TMDB non-200 response | HTTP 502 with TMDB status and body excerpt |
| Movie title not in local dataset | HTTP 404 with descriptive message |
| Missing pickle files at startup | `RuntimeError` — server will not start |
| Invalid home feed category | HTTP 400 |

The Streamlit frontend surfaces all backend errors via `st.error(...)` and never crashes silently.

---

## Known Limitations & Future Work

- **TF-IDF title resolution**: The local index uses exact (lowercased) title matching. Titles with subtitles, punctuation differences, or year suffixes will return 404 from `/recommend/tfidf` even when the movie exists in the dataset.
- **Cold-start latency**: Each TF-IDF recommendation requires a separate TMDB search call to fetch a poster. For `tfidf_top_n=12` this means up to 12 sequential TMDB requests. Batching or caching these would significantly reduce response time.
- **Single genre filtering**: Genre recommendations use only the first genre from `details.genres`. Weighted multi-genre discover queries would yield better diversity.
- **No user accounts or rating history**: Recommendations are purely content-based. Collaborative filtering (e.g. matrix factorisation on user–movie ratings) would personalise results further.
- **Streamlit session state**: Navigation state is not preserved across browser refreshes unless the `?view=` and `?id=` query parameters are present in the URL — which they are after a navigation, so deep-linking works correctly.