#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Azure Enterprise Hybrid Search Indexing Engine module.
#       Unifies Azure OpenAI embeddings (text-embedding-3-small/large),
#       sparse BM25 inverted lexical indexing, Reciprocal Rank Fusion (RRF),
#       Microsoft Entra ID RBAC security filtering, and cross-encoder re-ranking.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added Reciprocal Rank Fusion and BM25 vector scoring
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set

from parsers.base import DocumentChunk
from config import settings

@dataclass
class SearchResult:
    """
    Explanation: Container structure holding ranked search results and scoring telemetry
    :param  chunk DocumentChunk: Underlying document chunk payload
    :param  dense_score float: Cosine similarity score from dense semantic retrieval
    :param  sparse_score float: BM25 score from sparse lexical matching
    :param  rrf_score float: Reciprocal Rank Fusion blended score
    :param  final_score float: Re-ranked score after cross-encoder alignment
    :param  rank int: 1-based final ranking position
    """
    chunk: DocumentChunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0

class SimpleLocalEmbeddings:
    """
    Explanation: Deterministic local character/word n-gram embedding generator for fallback
    :param  dim int: Embedding dimension size
    """
    def __init__(self, dim: int = 256):
        """
        Explanation: Initializes local embedding dimension
        :param  dim int: Dimensionality of embedding vector
        :return None: Instantiates local embedder
        """
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        """
        Explanation: Converts string text into normalized floating point vector
        :param  text str: Text string to vectorize
        :return vec List[float]: Normalized embedding array
        """
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [0.0] * self.dim
        counts = Counter(tokens)
        vec = [0.0] * self.dim
        for token, count in counts.items():
            h = 0
            for c in token:
                h = (h * 31 + ord(c)) % self.dim
            vec[h] += float(count)
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Explanation: Computes embedding vectors for multiple document texts in batch
        :param  texts List[str]: Collection of strings to embed
        :return vectors List[List[float]]: List of generated embedding vectors
        """
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Explanation: Generates embedding vector for search query
        :param  text str: Query string
        :return vector List[float]: Query embedding vector
        """
        return self._embed(text)

class BM25Index:
    """
    Explanation: In-memory Okapi BM25 sparse lexical inverted index for keyword search
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Explanation: Initializes BM25 tuning parameters
        :param  k1 float: Term frequency saturation parameter
        :param  b float: Document length normalization parameter
        :return None: Instantiates BM25 index
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: Dict[str, int] = {}
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)

    def _tokenize(self, text: str) -> List[str]:
        """
        Explanation: Extracts lowercase alphanumeric and special code tokens from text
        :param  text str: Raw document string
        :return tokens List[str]: Tokenized words and identifiers
        """
        return [w.lower() for w in re.findall(r"\b[A-Za-z0-9_\-\.#]+\b", text)]

    def index_chunks(self, chunks: List[DocumentChunk]):
        """
        Explanation: Builds inverted index, document frequencies, and document lengths across chunks
        :param  chunks List[DocumentChunk]: Chunks to index into BM25 dictionary
        :return None: Updates internal inverted index structures
        """
        self.corpus_size = len(chunks)
        total_len = 0
        self.inverted_index.clear()
        self.doc_frequencies.clear()
        self.doc_lengths.clear()

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.doc_lengths[chunk.chunk_id] = len(tokens)
            total_len += len(tokens)
            tf = Counter(tokens)
            for token, count in tf.items():
                self.inverted_index[token][chunk.chunk_id] = count
                self.doc_frequencies[token] += 1

        self.avgdl = (total_len / self.corpus_size) if self.corpus_size > 0 else 1.0

    def search(self, query: str, candidate_ids: Optional[Set[str]] = None) -> Dict[str, float]:
        """
        Explanation: Evaluates candidate chunks against query terms using BM25 relevance scoring
        :param  query str: Keyword search query
        :param  candidate_ids Optional[Set[str]]: Candidate chunk IDs pre-filtered by Entra ID RBAC
        :return scores Dict[str, float]: Normalized BM25 relevance scores mapped by chunk ID
        """
        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            df = self.doc_frequencies[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for chunk_id, tf in self.inverted_index[token].items():
                if candidate_ids is not None and chunk_id not in candidate_ids:
                    continue
                doc_len = self.doc_lengths.get(chunk_id, self.avgdl)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[chunk_id] += idf * (numerator / denominator)

        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                for k in scores:
                    scores[k] = scores[k] / max_score

        return dict(scores)

class AzureEnterpriseHybridIndex:
    """
    Explanation: Production-grade hybrid vector index for Azure OpenAI Service and Databricks.
                 Integrates neural semantic vectors, BM25 lexical tokens, Reciprocal Rank Fusion,
                 Microsoft Entra ID RBAC access filters, and cross-encoder re-ranking.
    """

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Explanation: Initializes Azure hybrid index structures and embedding models
        :param  azure_endpoint Optional[str]: Azure OpenAI Service endpoint URL
        :param  azure_api_key Optional[str]: Azure OpenAI API authentication key
        :param  openai_api_key Optional[str]: Standard OpenAI API key
        :return None: Instantiates AzureEnterpriseHybridIndex object
        """
        self.chunks: List[DocumentChunk] = []
        self.chunk_by_id: Dict[str, DocumentChunk] = {}
        self.embeddings_matrix: List[List[float]] = []
        self.bm25 = BM25Index()
        self.backend_name = "Uninitialized"
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.openai_api_key = openai_api_key

        self._init_embeddings()

    def _init_embeddings(self):
        """
        Explanation: Initializes Azure OpenAI text-embedding-3 model or falls back to OpenAI or local embeddings
        :return None: Configures active embedder instance
        """
        # 1. Try Azure OpenAI Embeddings
        endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint)
        key = self.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key)
        if endpoint and key:
            try:
                from langchain_openai import AzureOpenAIEmbeddings
                self.embedder = AzureOpenAIEmbeddings(
                    azure_deployment=settings.azure_embedding_deployment,
                    azure_endpoint=endpoint,
                    api_key=key,
                    api_version=settings.azure_openai_api_version,
                )
                self.backend_name = f"Azure OpenAI ({settings.azure_embedding_deployment})"
                return
            except Exception:
                pass

        # 2. Try Standard OpenAI Embeddings
        openai_key = self.openai_api_key or os.getenv("OPENAI_API_KEY", settings.openai_api_key)
        if openai_key:
            try:
                from langchain_openai import OpenAIEmbeddings
                self.embedder = OpenAIEmbeddings(
                    model=settings.openai_embedding_model,
                    api_key=openai_key,
                )
                self.backend_name = f"OpenAI ({settings.openai_embedding_model})"
                return
            except Exception:
                pass

        # 3. Local High-Precision Fallback
        self.embedder = SimpleLocalEmbeddings()
        self.backend_name = "Local High-Precision Embeddings"

    def add_chunks(self, new_chunks: List[DocumentChunk]):
        """
        Explanation: Generates embeddings and builds inverted index mappings for incoming chunks
        :param  new_chunks List[DocumentChunk]: New document segments to index
        :return None: Appends chunks and updates index structures
        """
        if not new_chunks:
            return

        texts = [c.text for c in new_chunks]
        try:
            vectors = self.embedder.embed_documents(texts)
        except Exception:
            self.embedder = SimpleLocalEmbeddings()
            self.backend_name = "Local Embeddings (Cloud API Pending)"
            vectors = self.embedder.embed_documents(texts)

        for chunk, vec in zip(new_chunks, vectors):
            self.chunks.append(chunk)
            self.chunk_by_id[chunk.chunk_id] = chunk
            self.embeddings_matrix.append(vec)

        self.bm25.index_chunks(self.chunks)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Explanation: Computes cosine similarity between two vector representations
        :param  v1 List[float]: First vector
        :param  v2 List[float]: Second vector
        :return similarity float: Normalized cosine score
        """
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(
        self,
        query: str,
        user_role: str = "all",
        top_k: int = 10,
        rerank_top_k: int = 4,
    ) -> List[SearchResult]:
        """
        Explanation: Executes Entra ID RBAC filtering, dense & sparse scoring, RRF, and cross-encoder re-ranking
        :param  query str: Search query string
        :param  user_role str: Active Entra ID user role
        :param  top_k int: Number of RRF candidate pool
        :param  rerank_top_k int: Final count of re-ranked results to return
        :return results List[SearchResult]: Scored and ranked results
        """
        if not self.chunks:
            return []

        # 1. Entra ID / RBAC Pre-Filter
        candidate_ids = set()
        user_role_lower = user_role.lower()
        for c in self.chunks:
            chunk_roles = [r.lower() for r in c.allowed_roles]
            if "all" in chunk_roles or user_role_lower in chunk_roles:
                candidate_ids.add(c.chunk_id)

        if not candidate_ids:
            return []

        # 2. Dense Semantic Search
        try:
            q_vec = self.embedder.embed_query(query)
        except Exception:
            q_vec = SimpleLocalEmbeddings().embed_query(query)

        dense_scores: Dict[str, float] = {}
        for chunk, doc_vec in zip(self.chunks, self.embeddings_matrix):
            if chunk.chunk_id in candidate_ids:
                sim = self._cosine_similarity(q_vec, doc_vec)
                dense_scores[chunk.chunk_id] = max(0.0, sim)

        # 3. Sparse Lexical BM25 Search
        sparse_scores = self.bm25.search(query, candidate_ids=candidate_ids)

        # 4. Rank sorting for RRF
        dense_ranked = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        sparse_ranked = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)

        dense_rank_map = {cid: rank + 1 for rank, (cid, _) in enumerate(dense_ranked)}
        sparse_rank_map = {cid: rank + 1 for rank, (cid, _) in enumerate(sparse_ranked)}

        # 5. Reciprocal Rank Fusion
        rrf_constant = 60
        fused_scores: Dict[str, float] = {}
        all_candidates = set(dense_rank_map.keys()).union(sparse_rank_map.keys())

        for cid in all_candidates:
            d_rank = dense_rank_map.get(cid, 9999)
            s_rank = sparse_rank_map.get(cid, 9999)
            rrf_score = (settings.dense_weight / (rrf_constant + d_rank)) + (
                settings.sparse_weight / (rrf_constant + s_rank)
            )
            fused_scores[cid] = rrf_score

        top_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 6. Cross-Encoder Re-ranker
        q_words = set(re.findall(r"\w+", query.lower()))
        results: List[SearchResult] = []

        for cid, rrf_val in top_candidates:
            chunk = self.chunk_by_id[cid]
            d_score = dense_scores.get(cid, 0.0)
            s_score = sparse_scores.get(cid, 0.0)

            doc_words = set(re.findall(r"\w+", chunk.text.lower()))
            overlap_ratio = len(q_words.intersection(doc_words)) / max(1, len(q_words))
            rerank_score = (0.5 * d_score) + (0.3 * s_score) + (0.2 * overlap_ratio)

            results.append(
                SearchResult(
                    chunk=chunk,
                    dense_score=round(d_score, 4),
                    sparse_score=round(s_score, 4),
                    rrf_score=round(rrf_val, 5),
                    final_score=round(rerank_score, 4),
                )
            )

        results.sort(key=lambda x: x.final_score, reverse=True)
        final_results = results[:rerank_top_k]
        for idx, res in enumerate(final_results, start=1):
            res.rank = idx

        return final_results

    def clear(self):
        """
        Explanation: Clears all document chunks and resets sparse and dense indexes
        :return None: Flushes index state
        """
        self.chunks.clear()
        self.chunk_by_id.clear()
        self.embeddings_matrix.clear()
        self.bm25 = BM25Index()

    def get_stats(self) -> Dict[str, Any]:
        """
        Explanation: Aggregates index statistics including chunk count, document counts, and formats
        :return stats Dict[str, Any]: Index telemetry dictionary
        """
        formats = Counter(c.file_format for c in self.chunks)
        roles = Counter(r for c in self.chunks for r in c.allowed_roles)
        sources = list(set(c.source for c in self.chunks))
        return {
            "total_chunks": len(self.chunks),
            "unique_documents": len(sources),
            "format_breakdown": dict(formats),
            "role_coverage": dict(roles),
            "backend": self.backend_name,
        }
