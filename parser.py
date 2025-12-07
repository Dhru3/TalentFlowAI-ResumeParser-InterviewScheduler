"""
AI-powered Resume Parser (Batch + Cached)
Uses Groq for LLM extraction and HuggingFace sentence-transformers for embeddings.
"""

import os
import json
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import re
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None


# ==============================
# CONFIGURATION
# ==============================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 3  # Parse 3 resumes per API call

# Global embedding model instance (lazy loaded)
_embedding_model = None


# ==============================
# DATA STRUCTURE
# ==============================

@dataclass
class ResumeRecord:
    file_name: str
    text: str  # Full extracted resume text for ranking
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    skills: List[str]
    education: List[str]
    experience: List[str]
    summary: Optional[str]
    embedding: Optional[List[float]]


# ==============================
# UTILITIES
# ==============================

def extract_text_from_pdf(pdf_path: Path) -> str:
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text.strip())
    return "\n".join(text_parts)


def call_groq_llm_for_batch(resume_texts: List[str], file_names: List[str]) -> dict:
    """Send multiple resumes to Groq LLM in a single structured request."""
    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY not set")
        return []
    
    client = Groq(api_key=GROQ_API_KEY)

    prompt = (
        "You are a professional resume parser. Extract structured details for each resume below.\n\n"
        "CRITICAL: Return ONLY valid JSON with no markdown formatting, explanations, or extra text.\n"
        "Use double quotes for all strings. Do not use single quotes.\n\n"
        "Required JSON format:\n"
        '{\n'
        '  "results": [\n'
        '    {\n'
        '      "file": "filename.pdf",\n'
        '      "name": "Full Name or null",\n'
        '      "email": "email@example.com or null",\n'
        '      "phone": "Phone number or null",\n'
        '      "skills": ["skill1", "skill2"],\n'
        '      "education": ["degree info"],\n'
        '      "experience": ["job title and company"],\n'
        '      "summary": "Brief professional summary or null"\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "Now parse the following resumes:\n"
    )

    for i, text in enumerate(resume_texts):
        prompt += f"\n--- Resume {i+1}: {file_names[i]} ---\n{text[:12000]}\n"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a JSON-only resume parser. Return only valid JSON with no markdown code blocks or explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks if present
    if raw.startswith("```"):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
    
    try:
        data = json.loads(raw)
        return data.get("results", [])
    except json.JSONDecodeError as e:
        print(f"⚠️ Groq returned invalid JSON: {e}")
        print(f"Raw response (first 500 chars): {raw[:500]}")
        return []


# Global embedding model instance
_embedding_model = None

def _get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"✓ Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"⚠️ Failed to load embedding model: {e}")
            return None
    return _embedding_model


def compute_embedding_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Compute embeddings for a batch of texts using HuggingFace sentence-transformers."""
    model = _get_embedding_model()
    if model is None:
        return [None] * len(texts)

    embeddings = []
    for i, text in enumerate(texts):
        try:
            # Truncate text to reasonable length for embedding
            truncated = text[:8000] if len(text) > 8000 else text
            vector = model.encode(truncated, convert_to_numpy=True)
            # Normalize the vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            embeddings.append(vector.tolist())
            print(f"  ✓ Generated embedding {i+1}/{len(texts)}")
        except Exception as e:
            print(f"⚠️ Embedding error for text {i+1}: {e}")
            import traceback
            traceback.print_exc()
            embeddings.append(None)
    return embeddings


def load_from_cache(file_name: str) -> Optional[dict]:
    cache_path = CACHE_DIR / f"{file_name}.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return None


def save_to_cache(file_name: str, data: dict):
    cache_path = CACHE_DIR / f"{file_name}.json"
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


# ==============================
# MAIN PARSER CLASS
# ==============================

class ResumeParserLLM:
    def __init__(self, compute_embeddings: bool = False):
        self.compute_embeddings = compute_embeddings

    def parse_folder(self, folder_path: str) -> pd.DataFrame:
        records = []
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))

        pending_files = []
        pending_texts = []

        for pdf in pdf_files:
            cached = load_from_cache(pdf.stem)
            if cached:
                print(f"✅ Loaded from cache: {pdf.name}")
                records.append(cached)
                continue

            text = extract_text_from_pdf(pdf)
            pending_files.append(pdf.name)
            pending_texts.append(text)

            if len(pending_files) >= BATCH_SIZE:
                batch_results = self._process_batch(pending_texts, pending_files)
                records.extend(batch_results)
                pending_files, pending_texts = [], []

        # Handle leftover files
        if pending_files:
            batch_results = self._process_batch(pending_texts, pending_files)
            records.extend(batch_results)

        df = pd.DataFrame(records)
        return df
    
    def parse_folder_files(self, pdf_files: List[Path]) -> pd.DataFrame:
        """Parse specific PDF files (used for incremental updates)."""
        records = []
        pending_files = []
        pending_texts = []

        for pdf in pdf_files:
            cached = load_from_cache(pdf.stem)
            if cached:
                print(f"✅ Loaded from cache: {pdf.name}")
                records.append(cached)
                continue

            text = extract_text_from_pdf(pdf)
            pending_files.append(pdf.name)
            pending_texts.append(text)

            if len(pending_files) >= BATCH_SIZE:
                batch_results = self._process_batch(pending_texts, pending_files)
                records.extend(batch_results)
                pending_files, pending_texts = [], []

        # Handle leftover files
        if pending_files:
            batch_results = self._process_batch(pending_texts, pending_files)
            records.extend(batch_results)

        df = pd.DataFrame(records)
        return df

    def _process_batch(self, texts: List[str], files: List[str]) -> List[dict]:
        print(f"🔍 Sending batch of {len(files)} resumes to Groq LLM...")
        results = call_groq_llm_for_batch(texts, files)
        
        # If batch parsing failed and we have multiple resumes, try one-by-one
        if not results and len(files) > 1:
            print("⚠️ Batch parsing failed. Trying individual parsing...")
            results = []
            for text, fname in zip(texts, files):
                single_result = call_groq_llm_for_batch([text], [fname])
                if single_result:
                    results.extend(single_result)
                else:
                    # Create minimal record with just the filename and text
                    results.append({
                        "file": fname,
                        "name": None,
                        "email": None,
                        "phone": None,
                        "skills": [],
                        "education": [],
                        "experience": [],
                        "summary": None
                    })
        elif not results:
            # Create minimal records for failed batch
            print("⚠️ All parsing attempts failed. Creating minimal records...")
            results = [{
                "file": fname,
                "name": None,
                "email": None,
                "phone": None,
                "skills": [],
                "education": [],
                "experience": [],
                "summary": None
            } for fname in files]
        
        embeddings = compute_embedding_batch(texts) if self.compute_embeddings else [None] * len(texts)

        batch_data = []
        for i, res in enumerate(results):
            record = ResumeRecord(
                file_name=res.get("file", files[i]),
                text=texts[i],  # Store original extracted text
                name=res.get("name"),
                email=res.get("email"),
                phone=res.get("phone"),
                skills=res.get("skills", []),
                education=res.get("education", []),
                experience=res.get("experience", []),
                summary=res.get("summary"),
                embedding=embeddings[i],
            )
            save_to_cache(Path(files[i]).stem, asdict(record))
            batch_data.append(asdict(record))
        return batch_data


# ==============================
# USAGE EXAMPLE
# ==============================

if __name__ == "__main__":
    parser = ResumeParserLLM(compute_embeddings=True)
    df = parser.parse_folder("resumes/")
    df.to_csv("parsed_resumes.csv", index=False)
    print("✅ Parsing completed. Results saved to parsed_resumes.csv")
