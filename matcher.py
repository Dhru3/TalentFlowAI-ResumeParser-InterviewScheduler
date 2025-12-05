"""Match job descriptions against parsed resumes using Azure OpenAI embeddings."""
from __future__ import annotations

import os
import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Any, Callable

import numpy as np
import pandas as pd

try:  # Optional – allow import without Azure OpenAI installed.
    from openai import AzureOpenAI
except ImportError:  # pragma: no cover - runtime guard
    AzureOpenAI = None  # type: ignore

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9\+#]+")
_COMMON_SKILL_KEYWORDS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "aws",
    "gcp",
    "azure",
    "docker",
    "kubernetes",
    "ml",
    "ai",
    "data",
    "analytics",
    "analyst",
    "developer",
    "engineer",
    "manager",
    "project",
    "cloud",
    "devops",
    "sre",
    "architect",
    "security",
    "network",
    "testing",
    "qa",
    "excel",
    "tableau",
    "pandas",
    "nlp",
    "c",
    "c++",
    "c#",
    "go",
    "rust",
    "scala",
}

MAX_MATCH_SCORE = 94.0


def _ensure_numpy_array(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vector), dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Embedding vector must be one-dimensional")
    return arr


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a = _ensure_numpy_array(vec_a)
    b = _ensure_numpy_array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class MatchResult:
    resume_index: int
    score: float
    metadata: dict


class ResumeMatcher:
    """Rank resumes from a DataFrame against a job description."""

    def __init__(
        self,
        *,
        embedding_model: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        resolved_model = (
            embedding_model
            or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
            or "text-embedding-3-large"
        )
        self._embedding_model = resolved_model
        self._azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
        self._azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self._client: Optional[Any] = None
        self._client_factory = client_factory
        if AzureOpenAI is None and client_factory is None:  # pragma: no cover - runtime guard
            logger.warning(
                "openai package not installed – matcher embeddings unavailable."
            )

    def rank(
        self,
        job_description: str,
        resumes_df: pd.DataFrame,
        *,
        top_k: int = 10,
        text_column: str = "text",
        reference_resumes_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rank resumes against a job description, optionally using reference resumes.
        
        Args:
            job_description: The job description text
            resumes_df: DataFrame of candidate resumes to rank
            top_k: Number of top results to return
            text_column: Column name containing resume text
            reference_resumes_df: Optional DataFrame of reference resumes from previously hired candidates
        
        Returns:
            DataFrame with match scores, sorted by best match
        """
        if text_column not in resumes_df.columns:
            raise KeyError(f"DataFrame missing required column: {text_column}")
        if resumes_df.empty:
            return resumes_df.assign(match_score=pd.Series(dtype=float)).head(0)

        client = self._ensure_client()
        jd_embedding = self._embed_text(client, job_description)

        resume_texts = resumes_df[text_column].fillna("").astype(str).tolist()
        resume_embeddings = self._maybe_get_cached_embeddings(resumes_df)
        if resume_embeddings is None:
            resume_embeddings = self._embed_batch(client, resume_texts)

        # Raw cosine similarity plus lightweight skill bonus (no normalization)
        base_scores = [cosine_similarity(jd_embedding, emb) for emb in resume_embeddings]
        skill_bonus = self._compute_skill_bonus(job_description, resume_texts)
        adjusted_scores = [base + bonus for base, bonus in zip(base_scores, skill_bonus)]
        final_scores = self._scale_scores(adjusted_scores, cap=MAX_MATCH_SCORE)

        results_df = resumes_df.copy()
        results_df["raw_match_score"] = base_scores
        results_df["adjusted_match_score"] = adjusted_scores
        results_df["match_score"] = final_scores
        results_df.sort_values(by="match_score", ascending=False, inplace=True)
        return results_df.head(top_k).reset_index(drop=True)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        if AzureOpenAI is None:
            raise RuntimeError(
                "openai package is required for Azure OpenAI embeddings."
            )
        if not self._azure_api_key or not self._azure_endpoint:
            raise RuntimeError(
                "Azure OpenAI credentials missing. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
            )
        self._client = AzureOpenAI(
            api_key=self._azure_api_key,
            api_version=self._azure_api_version,
            azure_endpoint=self._azure_endpoint,
        )
        return self._client

    def _embed_text(self, client: Any, text: str) -> np.ndarray:
        response = client.embeddings.create(
            input=text,
            model=self._embedding_model,
        )
        if not response.data:
            raise RuntimeError("Azure OpenAI returned no embedding data")
        return _ensure_numpy_array(response.data[0].embedding)

    def _embed_batch(self, client: Any, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        response = client.embeddings.create(
            input=texts,
            model=self._embedding_model,
        )
        if not response.data:
            return [np.zeros(1, dtype=np.float32) for _ in texts]
        return [_ensure_numpy_array(item.embedding) for item in response.data]

    @staticmethod
    def _maybe_get_cached_embeddings(df: pd.DataFrame) -> Optional[list[np.ndarray]]:
        if "embedding" not in df.columns:
            return None
        embeddings: list[np.ndarray] = []
        for value in df["embedding"]:
            if value is None:
                return None
            if isinstance(value, str) and not value.strip():
                return None
            if not isinstance(value, (list, tuple, np.ndarray)):
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            try:
                embeddings.append(_ensure_numpy_array(value))
            except Exception:
                return None
        return embeddings

    def _compute_reference_bonus(
        self, candidate_embeddings: list[np.ndarray], reference_embeddings: list[np.ndarray]
    ) -> list[float]:
        """
        Compute bonus scores based on similarity to reference resumes (previously hired candidates).
        
        For each candidate, calculate the maximum similarity to any reference resume,
        then apply a weighted bonus (0.15 weight to encourage similar profiles).
        
        Args:
            candidate_embeddings: List of candidate resume embeddings
            reference_embeddings: List of reference resume embeddings from hired candidates
            
        Returns:
            List of bonus scores for each candidate
        """
        bonuses: list[float] = []
        for candidate_emb in candidate_embeddings:
            # Find maximum similarity to any reference resume
            max_ref_similarity = max(
                cosine_similarity(candidate_emb, ref_emb) 
                for ref_emb in reference_embeddings
            )
            # Weight the reference similarity (0.15 = 15% contribution)
            bonus = 0.15 * max_ref_similarity
            bonuses.append(bonus)
        return bonuses

    def _compute_skill_bonus(
        self, job_description: str, resume_texts: list[str]
    ) -> list[float]:
        job_tokens = set(self._tokenize(job_description))
        job_skill_targets = job_tokens & _COMMON_SKILL_KEYWORDS
        bonuses: list[float] = []
        for text in resume_texts:
            tokens = self._tokenize(text)
            generic_hits = sum(1 for token in tokens if token in _COMMON_SKILL_KEYWORDS)
            targeted_hits = (
                sum(1 for token in tokens if token in job_skill_targets)
                if job_skill_targets
                else 0
            )
            generic_only = max(generic_hits - targeted_hits, 0)
            bonus = 0.02 * targeted_hits + 0.005 * generic_only
            bonuses.append(bonus)
        return bonuses

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []
        return _TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _scale_scores(
        scores: list[float], *, cap: float = MAX_MATCH_SCORE, softness: float = 1.45
    ) -> list[float]:
        """Compress unbounded similarity scores into a capped 0-100 range."""
        if not scores:
            return []
        arr = np.asarray(scores, dtype=np.float32)
        clipped = np.clip(arr, -1.0, 1.0)
        normalized = (clipped + 1.0) * 0.5  # Map [-1, 1] -> [0, 1]
        softened = np.power(normalized, softness)  # Reduce generosity near the top
        scaled = softened * cap
        return scaled.tolist()


__all__ = ["ResumeMatcher", "MatchResult", "cosine_similarity", "MAX_MATCH_SCORE"]
