"""
Internal Talent Pool Manager with FAISS-based persistent embeddings store.

Features:
- Persistent FAISS index for fast similarity search
- Incremental updates (only parse new PDFs)
- Metadata tracking (filename, parse date, resume details)
- Efficient storage and retrieval
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import faiss

from parser import ResumeParserLLM

logger = logging.getLogger(__name__)


class InternalTalentPool:
    """Manages internal candidate resumes with persistent FAISS embeddings store."""
    
    def __init__(self, data_folder: Path = Path("data"), store_folder: Path = Path("internal_talent_store")):
        """
        Initialize the internal talent pool manager.
        
        Args:
            data_folder: Folder containing PDF resumes
            store_folder: Folder to store FAISS index and metadata
        """
        self.data_folder = Path(data_folder)
        self.store_folder = Path(store_folder)
        self.store_folder.mkdir(exist_ok=True)
        
        # File paths
        self.index_path = self.store_folder / "faiss_index.bin"
        self.metadata_path = self.store_folder / "metadata.pkl"
        self.indexed_files_path = self.store_folder / "indexed_files.json"
        
        # In-memory data
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.indexed_files: Dict[str, str] = {}  # filename -> last_modified_time
        
        # Load existing data
        self._load_store()
    
    def _load_store(self):
        """Load existing FAISS index and metadata from disk."""
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            
            if self.metadata_path.exists():
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"✅ Loaded metadata for {len(self.metadata)} candidates")
            
            if self.indexed_files_path.exists():
                with open(self.indexed_files_path, "r") as f:
                    self.indexed_files = json.load(f)
                logger.info(f"✅ Loaded index tracking for {len(self.indexed_files)} files")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load existing store: {e}")
            self.index = None
            self.metadata = []
            self.indexed_files = {}
    
    def _save_store(self):
        """Save FAISS index and metadata to disk."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
            
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            
            with open(self.indexed_files_path, "w") as f:
                json.dump(self.indexed_files, f, indent=2)
            
            logger.info("✅ Saved FAISS store to disk")
        except Exception as e:
            logger.error(f"❌ Failed to save store: {e}")
    
    def _get_file_modified_time(self, filepath: Path) -> str:
        """Get last modified time as string."""
        return str(filepath.stat().st_mtime)
    
    def _get_new_or_modified_files(self) -> List[Path]:
        """Get list of PDF files that are new or have been modified."""
        if not self.data_folder.exists():
            return []
        
        pdf_files = list(self.data_folder.glob("*.pdf"))
        new_files = []
        
        for pdf_file in pdf_files:
            filename = pdf_file.name
            current_mtime = self._get_file_modified_time(pdf_file)
            
            # Check if file is new or modified
            if filename not in self.indexed_files or self.indexed_files[filename] != current_mtime:
                new_files.append(pdf_file)
        
        return new_files
    
    def get_total_candidates(self) -> int:
        """Get total number of indexed candidates."""
        return len(self.metadata)
    
    def needs_update(self) -> bool:
        """Check if there are new or modified files to process."""
        new_files = self._get_new_or_modified_files()
        return len(new_files) > 0
    
    def update_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Update the FAISS index with new or modified resumes.
        
        Args:
            force_rebuild: If True, rebuild entire index from scratch
            
        Returns:
            Dictionary with update statistics
        """
        stats = {
            "total_files": 0,
            "new_files": 0,
            "updated_files": 0,
            "errors": 0,
            "success": False
        }
        
        if not self.data_folder.exists():
            return stats
        
        pdf_files = list(self.data_folder.glob("*.pdf"))
        stats["total_files"] = len(pdf_files)
        
        if force_rebuild:
            # Clear existing data
            self.index = None
            self.metadata = []
            self.indexed_files = {}
            files_to_process = pdf_files
        else:
            # Only process new/modified files
            files_to_process = self._get_new_or_modified_files()
        
        if not files_to_process:
            stats["success"] = True
            return stats
        
        logger.info(f"📊 Processing {len(files_to_process)} files...")
        
        # Parse resumes
        try:
            parser = ResumeParserLLM(compute_embeddings=True)
            new_df = parser.parse_folder_files(files_to_process)
            
            if new_df.empty:
                logger.warning("⚠️ No resumes parsed successfully")
                return stats
            
            # Extract embeddings and metadata
            new_embeddings = []
            new_metadata = []
            
            for idx, row in new_df.iterrows():
                embedding = row.get("Embedding")
                file_name = row.get("File Name", "Unknown")
                if embedding is None or not isinstance(embedding, (list, np.ndarray)):
                    logger.warning(f"⚠️ No valid embedding for {file_name} (type: {type(embedding)})")
                    stats["errors"] += 1
                    continue
                
                # Convert to numpy array and normalize
                emb_array = np.array(embedding, dtype=np.float32)
                if len(emb_array.shape) == 1:
                    emb_array = emb_array.reshape(1, -1)
                
                # Normalize for cosine similarity
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                
                new_embeddings.append(emb_array.flatten())
                
                # Store metadata
                metadata = {
                    "file_name": row.get("File Name", "Unknown"),
                    "name": row.get("Name"),
                    "email": row.get("Email"),
                    "phone": row.get("Phone"),
                    "skills": row.get("Skills", []),
                    "education": row.get("Education", []),
                    "experience": row.get("Experience", []),
                    "summary": row.get("Summary"),
                    "indexed_at": datetime.now().isoformat(),
                    "file_path": str(self.data_folder / row.get("File Name", "Unknown"))
                }
                new_metadata.append(metadata)
                
                # Track indexed file
                pdf_path = self.data_folder / row.get("File Name", "Unknown")
                if pdf_path.exists():
                    self.indexed_files[row.get("File Name")] = self._get_file_modified_time(pdf_path)
                    stats["new_files"] += 1
            
            if not new_embeddings:
                logger.warning("⚠️ No valid embeddings generated")
                return stats
            
            # Convert to numpy array
            new_embeddings_array = np.vstack(new_embeddings).astype(np.float32)
            
            # Create or update FAISS index
            if self.index is None:
                # Create new index
                dimension = new_embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
                logger.info(f"📊 Created new FAISS index (dimension: {dimension})")
            
            # Add to index
            self.index.add(new_embeddings_array)
            self.metadata.extend(new_metadata)
            
            # Save to disk
            self._save_store()
            
            stats["success"] = True
            logger.info(f"✅ Updated index: {len(new_embeddings)} new candidates indexed")
            
        except Exception as e:
            logger.error(f"❌ Failed to update index: {e}")
            stats["errors"] += 1
        
        return stats
    
    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> pd.DataFrame:
        """
        Search for similar candidates using query embedding.
        
        Args:
            query_embedding: Query embedding vector (normalized)
            top_k: Number of top results to return
            
        Returns:
            DataFrame with candidate details and similarity scores
        """
        if self.index is None or len(self.metadata) == 0:
            return pd.DataFrame()
        
        # Ensure query is normalized and correct shape
        query = np.array(query_embedding, dtype=np.float32)
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, k)
        
        # Build results DataFrame
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                meta["similarity_score"] = float(dist)
                meta["rank"] = i + 1
                results.append(meta)
        
        return pd.DataFrame(results)
    
    def get_all_candidates_df(self) -> pd.DataFrame:
        """Get DataFrame with all indexed candidates."""
        if not self.metadata:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metadata)
    
    def clear_store(self):
        """Clear all indexed data (useful for debugging/reset)."""
        self.index = None
        self.metadata = []
        self.indexed_files = {}
        
        # Remove files
        for path in [self.index_path, self.metadata_path, self.indexed_files_path]:
            if path.exists():
                path.unlink()
        
        logger.info("🗑️ Cleared internal talent pool store")
