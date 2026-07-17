#!/usr/bin/env python3
"""
Semantic Memory Search Module for Hermes + Open WebUI Integration
Fixes: https://github.com/open-webui/open-webui/issues/27079

Problem: Open WebUI searches memories with substring matching instead of semantic similarity.
Solution: Add embedding-based semantic search layer.

Usage:
  from memory_search import SemanticMemoryEngine
  engine = SemanticMemoryEngine()
  results = engine.search("What does the user like to drink?")
  # Returns: [Memory(text="user prefers coffee over tea", score=0.92)]
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️  sentence-transformers not installed. Using fallback substring search.")
    print("   Install: pip install sentence-transformers")


class MemoryType(Enum):
    PERSON = "person"  # Team member (Dr. Svensson, Lawyer Andersson)
    DEVICE = "device"  # PC config, Cisco router state
    SKILL = "skill"  # Learned patterns, successful approaches
    PATTERN = "pattern"  # Task patterns, workflows
    PREFERENCE = "preference"  # User likes/dislikes
    INSTRUCTION = "instruction"  # Standing instructions


@dataclass
class Memory:
    """A single memory with metadata"""
    id: str
    text: str
    memory_type: MemoryType
    created_at: str
    embedding: Optional[List[float]] = None
    relevance_score: float = 1.0
    tags: List[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.memory_type.value,
            "created_at": self.created_at,
            "relevance_score": round(self.relevance_score, 3),
            "tags": self.tags or []
        }


class SemanticMemoryEngine:
    """
    Semantic memory search for Hermes/Emma.
    Replaces Open WebUI's substring matching with embedding-based similarity.
    """
    
    def __init__(self, db_path: str = "~/hermes-homeserver/data/memories.db",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic memory engine.
        
        Args:
            db_path: SQLite database for memory storage
            model_name: Sentence-transformer model (all-MiniLM-L6-v2 = fast, mobile-friendly)
        """
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.similarity_threshold = 0.5  # Min relevance score (0-1)
        
        if HAS_EMBEDDINGS:
            try:
                print(f"🔄 Loading semantic model: {model_name}...")
                self.model = SentenceTransformer(model_name)
                print("✅ Semantic model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                HAS_EMBEDDINGS = False
        
        self._init_db()
    
    def _init_db(self):
        """Create memories table with embedding storage"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            memory_type TEXT,
            created_at TIMESTAMP,
            embedding BLOB,
            tags TEXT,
            relevance_score FLOAT DEFAULT 1.0
        )''')
        
        c.execute('''CREATE INDEX IF NOT EXISTS idx_memory_type 
                     ON memories(memory_type)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_created_at 
                     ON memories(created_at)''')
        
        conn.commit()
        conn.close()
    
    def add_memory(self, text: str, memory_type: MemoryType, tags: List[str] = None) -> str:
        """
        Add a memory to the knowledge base.
        
        Example:
            engine.add_memory(
                "Dr. Svensson handles healthcare queries and reports",
                MemoryType.PERSON,
                tags=["doctor", "healthcare", "sweden"]
            )
        """
        memory_id = hashlib.sha256(
            f"{text}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Compute embedding if available
        embedding = None
        if HAS_EMBEDDINGS:
            embedding = self._encode(text)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        embedding_blob = None
        if embedding:
            import numpy as np
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        
        c.execute('''INSERT INTO memories 
                     (id, text, memory_type, created_at, embedding, tags)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (memory_id, text, memory_type.value, datetime.now(),
                   embedding_blob, json.dumps(tags or [])))
        
        conn.commit()
        conn.close()
        
        return memory_id
    
    def _encode(self, text: str) -> List[float]:
        """Convert text to embedding vector"""
        if not HAS_EMBEDDINGS or not self.model:
            return None
        return self.model.encode(text, convert_to_tensor=False).tolist()
    
    def search(self, query: str, memory_type: MemoryType = None, 
               top_k: int = 5) -> List[Memory]:
        """
        Semantic search over memories.
        
        Args:
            query: Natural language search (e.g., "What does user like to drink?")
            memory_type: Filter by type (optional)
            top_k: Return top K results
        
        Returns:
            List of Memory objects ranked by relevance
        
        Example:
            results = engine.search("What's the user's favorite drink?")
            for mem in results:
                print(f"{mem.text} (score: {mem.relevance_score:.1%})")
        """
        
        if not HAS_EMBEDDINGS or not self.model:
            # Fallback: substring matching
            return self._search_substring(query, memory_type, top_k)
        
        # Semantic search
        query_embedding = self._encode(query)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Fetch all memories (or filtered by type)
        if memory_type:
            c.execute("SELECT id, text, memory_type, created_at, embedding, tags FROM memories WHERE memory_type = ?",
                      (memory_type.value,))
        else:
            c.execute("SELECT id, text, memory_type, created_at, embedding, tags FROM memories")
        
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Compute similarity scores
        results = []
        for row in rows:
            mem_id, text, mem_type, created_at, embedding_blob, tags_json = row
            
            if embedding_blob:
                import numpy as np
                memory_embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
                
                # Compute cosine similarity
                similarity = util.cos_sim(
                    [query_embedding], 
                    [memory_embedding]
                )[0][0].item()
                
                if similarity >= self.similarity_threshold:
                    results.append(Memory(
                        id=mem_id,
                        text=text,
                        memory_type=MemoryType(mem_type),
                        created_at=created_at,
                        embedding=memory_embedding,
                        relevance_score=float(similarity),
                        tags=json.loads(tags_json or "[]")
                    ))
        
        # Sort by relevance (highest first)
        results.sort(key=lambda m: m.relevance_score, reverse=True)
        return results[:top_k]
    
    def _search_substring(self, query: str, memory_type: MemoryType = None, 
                          top_k: int = 5) -> List[Memory]:
        """Fallback: naive substring search"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        query_lower = query.lower()
        
        if memory_type:
            c.execute("SELECT id, text, memory_type, created_at, tags FROM memories WHERE memory_type = ?",
                      (memory_type.value,))
        else:
            c.execute("SELECT id, text, memory_type, created_at, tags FROM memories")
        
        rows = c.fetchall()
        conn.close()
        
        # Score by substring match
        results = []
        for row in rows:
            mem_id, text, mem_type, created_at, tags_json = row
            text_lower = text.lower()
            
            # Simple scoring: number of matching words
            query_words = set(query_lower.split())
            matched_words = sum(1 for word in query_words if word in text_lower)
            score = matched_words / len(query_words) if query_words else 0
            
            if score > 0:
                results.append(Memory(
                    id=mem_id,
                    text=text,
                    memory_type=MemoryType(mem_type),
                    created_at=created_at,
                    relevance_score=score,
                    tags=json.loads(tags_json or "[]")
                ))
        
        results.sort(key=lambda m: m.relevance_score, reverse=True)
        return results[:top_k]
    
    def export_for_open_webui(self, memories: List[Memory]) -> Dict:
        """
        Format memories for Open WebUI integration.
        
        Returns dict compatible with Open WebUI's memory format.
        """
        return {
            "memories": [mem.to_dict() for mem in memories],
            "timestamp": datetime.now().isoformat(),
            "engine": "semantic",
            "model": self.model_name
        }


# ============================================================================
# INTEGRATION WITH OPEN WEBUI
# ============================================================================

class OpenWebUIMemoryBridge:
    """
    Bridge between Hermes semantic memory and Open WebUI.
    
    Usage in Open WebUI:
    1. Hook into memory search in web/models/memory.ts
    2. Call this bridge instead of substring search
    """
    
    def __init__(self):
        self.engine = SemanticMemoryEngine()
    
    def search_memories(self, query: str, user_id: str = None) -> List[Dict]:
        """
        Open WebUI compatible memory search.
        Replaces the existing search_memories function.
        """
        results = self.engine.search(query, top_k=10)
        return [mem.to_dict() for mem in results]
    
    def add_memory_from_conversation(self, text: str, message_content: str) -> str:
        """
        Auto-extract and add memory from user message.
        
        Example: User says "I prefer coffee over tea"
        → Automatically creates PREFERENCE memory
        """
        # Simple heuristics to detect memory type
        if any(word in text.lower() for word in ["prefer", "like", "love", "hate", "dislike"]):
            mem_type = MemoryType.PREFERENCE
        elif any(word in text.lower() for word in ["person", "team", "member", "colleague", "doctor", "lawyer"]):
            mem_type = MemoryType.PERSON
        elif any(word in text.lower() for word in ["pc", "device", "computer", "server", "router"]):
            mem_type = MemoryType.DEVICE
        else:
            mem_type = MemoryType.INSTRUCTION
        
        return self.engine.add_memory(text, mem_type)


# ============================================================================
# EMMA TTS INTEGRATION
# ============================================================================

class EmmaWithSemanticMemory:
    """Emma co-producer with semantic memory search"""
    
    def __init__(self):
        self.memory_engine = SemanticMemoryEngine()
        self.tts_enabled = True
    
    def answer_with_memory(self, query: str) -> str:
        """
        Answer user queries by consulting semantic memory first.
        
        Example:
            query: "What does the user like to drink?"
            memory search finds: "user prefers coffee over tea"
            response: "The user prefers coffee over tea."
        """
        # Search memories semantically
        memories = self.memory_engine.search(query, top_k=3)
        
        if not memories:
            return "I don't have that information in my memory."
        
        # Build response from most relevant memory
        best_memory = memories[0]
        response = f"Based on my memory: {best_memory.text}"
        
        if self.tts_enabled:
            self._speak(response)
        
        return response
    
    def _speak(self, text: str):
        """Text-to-speech (Swedish + English)"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except ImportError:
            pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("🧠 Semantic Memory Engine - Hermes Integration")
    print("=" * 60)
    
    engine = SemanticMemoryEngine()
    
    # Add some memories
    print("\n📝 Adding memories...")
    engine.add_memory(
        "The user prefers coffee over tea",
        MemoryType.PREFERENCE,
        tags=["drink", "preference", "coffee"]
    )
    engine.add_memory(
        "User's favorite color is blue",
        MemoryType.PREFERENCE,
        tags=["color", "preference"]
    )
    engine.add_memory(
        "Dr. Svensson is the user's healthcare advisor",
        MemoryType.PERSON,
        tags=["doctor", "healthcare", "sweden"]
    )
    engine.add_memory(
        "PC-1 runs Ubuntu with web server role",
        MemoryType.DEVICE,
        tags=["pc-1", "ubuntu", "webserver"]
    )
    
    # Test semantic search
    print("\n🔍 Testing semantic search...")
    print("\nQuery: 'What does the user like to drink?'")
    results = engine.search("What does the user like to drink?")
    for mem in results:
        print(f"  ✓ {mem.text} (relevance: {mem.relevance_score:.1%})")
    
    print("\nQuery: 'Who is the doctor?'")
    results = engine.search("Who is the doctor?")
    for mem in results:
        print(f"  ✓ {mem.text} (relevance: {mem.relevance_score:.1%})")
    
    print("\nQuery: 'What servers do we have?'")
    results = engine.search("What servers do we have?")
    for mem in results:
        print(f"  ✓ {mem.text} (relevance: {mem.relevance_score:.1%})")
    
    # Test Emma integration
    print("\n\n🎤 Testing Emma with semantic memory...")
    emma = EmmaWithSemanticMemory()
    print("\nEmma's answer to 'What does the user like to drink?':")
    answer = emma.answer_with_memory("What does the user like to drink?")
    print(f"  {answer}")
