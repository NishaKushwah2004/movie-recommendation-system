import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


#  .env, cors config
load_dotenv()
TMBD_API_KEY = os.getenv("TMDB_API_KEY")    


TMDB_BASE = "https://api.themoviedb.org/3" 
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


if not TMBD_API_KEY:
    raise ValueError("TMDB_API_KEY is not set in environment variables")



app = FastAPI(title="Movie Recommendation API", description="API for movie recommendations based on user preferences and TMDB data", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# PATH and GLOBAL VARS configs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR,"indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR,"tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR,"tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


#  MODELS
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    vote_average: Optional[float] = None

class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[str] = []

class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb : Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tmdb_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


#  UTILITY FUNCTIONS
def _norm_title(t: str) -> str:
    return str(t).lower().strip()

def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{IMAGE_BASE}{path}"

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    safe TMDB get:
    Network errors -> 502
    TMDB API errors -> 502 with TMDB error message
    """

    q = dict(params)
    q["api_key"] = TMBD_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"TMDB Network error {type(e).__name__}: {str(e)}")
    
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB API error {r.status_code}: {r.text}")  

    return r.json()  

async def tmdb_card_from_results(results: List[dict], limit: int=20) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []

    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average") 
            )
        )

    return out

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", params={"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or data.get("name") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )

async def tmdb_search_movies(query: str, page: int=1) -> Dict[str, Any]:
    """Search TMDB for movies matching the query. Returns raw TMDB response.
    Streamlit will use this to get TMDB IDs for the search results, then fetch details for each movie separately.
    """
    return await tmdb_get(
        "/search/movie", 
        params={"query": query, "page": page, "language": "en-US", "include_adult": False})

async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query = query, page=1)
    results = data.get("results", [])
    return results[0] if results else None



def build_title_to_idx_map(indices: Any) -> Dict[str,int]:
    """
    indices.pkl can be:
    - dict(title -> idx)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """

    title_to_idx: Dict[str,int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    
    #  pandas series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("indices.pkl must be dict or pandas series -like (with .items())")

def get_local_idx_by_title(title: str) -> int:

    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    
    key = _norm_title(title)
    if key not in TITLE_TO_IDX:
        raise HTTPException(status_code=404, detail=f"Movie title '{title}' not found in local dataset")
    return int(TITLE_TO_IDX[key])

def tfidf_recommend_titles(
        query_title: str, top_n: int=10
) -> List[Tuple[str, float]]: 
    """ 
    Returns list of (title, score) from local df using cosine similarityor TF-IDF matrix.
    Safe against missing columns/rows.
    """

    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF data not initialized or tfidf resources not loaded ")
    
    idx = get_local_idx_by_title(query_title)
    
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()


    order = np.argsort(-scores)

    out: List[Tuple[str, float]]= []
    for i in order:
        if int(i) == int (idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out

async def attach_tmdb_card_by_title(title:str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        n = await tmdb_search_first(title)
        if not n:
            return None
        return TMDBMovieCard(
            tmdb_id=int(n["id"]),
            title=n.get("title") or title,
            release_date=n.get("release_date"),
            poster_url=make_img_url(n.get("poster_path")),
            vote_average=n.get("vote_average"),
        )
    except Exception:
        return None
    

#  STARTUP : LOAD PICKELS
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)
    
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a dataframe with a 'title' column")


#  Routes
@app.get("/health")
def health():
    return {"status": "ok"}


#  HOME ROUTE
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str =Query("popular"),
    limit: int = Query(24, ge=1, le=100),
):
    """
    Home feed for Steamlit (posters).
    category:
    - trending (trending/movie/day)
    - popular, top_rated, upcoming, now_playing (movie/{category})
    """

    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", params={"language": "en-US"})
            return await tmdb_card_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail=f"Invalid category '{category}'")
        
        data = await tmdb_get(f"/movie/{category}", params={"language": "en-US", "page": 1})
        return await tmdb_card_from_results(data.get("results", []), limit=limit)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route Failed: {e}")


#  SEARCH MULTIPLE VIA KEYWORD   
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=1000),
):
    """
    Returns RAW TMDB shape with 'results' list.
    Streamlit will use it for:
    - dropdown suggestions
    - grid results
    """

    return await tmdb_search_movies(query=query, page=page)


#  MOVIE DETAILS
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_result(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


#  GENRE RECOMMENDATION
@app.get("/recommend/genre",response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50)
):
    """
    Given a TMDB movie ID:
    - fetch details
    - pick first genre
    - discover movies in that genre (popular)
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []
    
    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        params={"with_genres": genre_id, "sort_by": "popularity.desc", "language": "en-US", "page": 1},
    )

    cards = await tmdb_card_from_results(discover.get("/results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


#  TF-IDF RECOMMENDATION
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50)
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score":s} for t, s in recs]


#  BUNDLE: DETAILS + TFIDF + GENRE
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_list: int = Query(12, ge=1, le=30)
):
    """
    This endpoint is for whne you have a selected movie and want:
    - movie details
    - tf-idf recommendations(local) + posters
    - genre recommendations (TMDB) + posters
    
    NOTE:
    - It selects the BEST match from TMDB for the given query
    - If you want MULTIPLE matches, use /tmdb/search"""

    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB match found for query '{query}'")
    
    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id) 

    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []

    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except HTTPException:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except HTTPException:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    #  2) genre recommendations (tmdb discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            params={"with_genres": genre_id, "sort_by": "popularity.desc", "language": "en-US", "page": 1},
        )
        cards = await tmdb_card_from_results(discover.get("results", []), limit=genre_list)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tmdb_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


