# Memory and session compaction system
# Keeps long-running work usable by compressing context over time

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class MemorySegment:
    """A segment of memory that can be compacted."""
    segment_id: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance_score: float = 1.0
    is_compacted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def record_access(self):
        """Record an access to this segment."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def update_importance(self, score: float):
        """Update the importance score."""
        self.importance_score = max(0.0, min(1.0, score))


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    original_count: int
    compacted_count: int
    space_saved: float  # Percentage
    compacted_segments: List[str]
    summary: str


class MemoryCompactor:
    """
    Handles compaction of working memory and episodic memories.
    
    Compaction strategies:
    1. Summarize low-importance memories
    2. Merge similar memories
    3. Archive old memories
    4. Prune stale memories
    """
    
    def __init__(self, max_segments: int = 100, compaction_threshold: float = 0.3):
        self.max_segments = max_segments
        self.compaction_threshold = compaction_threshold
    
    def compact_working_memory(
        self,
        segments: List[MemorySegment],
        llm_summarizer: Optional[Any] = None
    ) -> CompactionResult:
        """
        Compact working memory by summarizing low-importance segments.
        
        Args:
            segments: List of memory segments
            llm_summarizer: Optional LLM to generate summaries
        
        Returns:
            CompactionResult with summary of changes
        """
        original_count = len(segments)
        compacted_segments = []
        to_summarize = []
        
        # Sort by importance and access recency
        sorted_segments = sorted(
            segments,
            key=lambda s: (s.importance_score, s.last_accessed),
            reverse=True
        )
        
        # Keep high-importance segments as-is
        for segment in sorted_segments:
            if segment.importance_score >= self.compaction_threshold:
                compacted_segments.append(segment)
            else:
                to_summarize.append(segment)
        
        # Summarize low-importance segments
        if to_summarize and llm_summarizer:
            summary_text = self._generate_summary(to_summarize, llm_summarizer)
            summary_segment = MemorySegment(
                segment_id=f"summary_{len(compacted_segments)}",
                content=summary_text,
                importance_score=self.compaction_threshold,
                is_compacted=True,
                metadata={"original_count": len(to_summarize)}
            )
            compacted_segments.append(summary_segment)
        elif to_summarize:
            # Without LLM, just keep the most recent
            compacted_segments.extend(to_summarize[-5:])
        
        compacted_count = len(compacted_segments)
        space_saved = ((original_count - compacted_count) / original_count * 100) if original_count > 0 else 0
        
        return CompactionResult(
            original_count=original_count,
            compacted_count=compacted_count,
            space_saved=space_saved,
            compacted_segments=[s.segment_id for s in to_summarize],
            summary=f"Compacted {original_count} -> {compacted_count} segments ({space_saved:.1f}% saved)"
        )
    
    def compact_episodic_memories(
        self,
        memories: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> CompactionResult:
        """
        Compact episodic memories by merging similar ones.
        
        Args:
            memories: List of episodic memory dicts
            similarity_threshold: Threshold for considering memories similar
        
        Returns:
            CompactionResult with summary of changes
        """
        original_count = len(memories)
        merged_groups = []
        used_indices = set()
        
        # Group similar memories
        for i, mem1 in enumerate(memories):
            if i in used_indices:
                continue
            
            group = [mem1]
            used_indices.add(i)
            
            for j, mem2 in enumerate(memories):
                if j in used_indices:
                    continue
                
                similarity = self._compute_similarity(mem1, mem2)
                if similarity >= similarity_threshold:
                    group.append(mem2)
                    used_indices.add(j)
            
            merged_groups.append(group)
        
        # Merge each group into a single memory
        compacted_memories = []
        compacted_ids = []
        
        for group in merged_groups:
            if len(group) > 1:
                merged = self._merge_memories(group)
                compacted_memories.append(merged)
                compacted_ids.extend([m.get("id", "unknown") for m in group])
            else:
                compacted_memories.append(group[0])
        
        compacted_count = len(compacted_memories)
        space_saved = ((original_count - compacted_count) / original_count * 100) if original_count > 0 else 0
        
        return CompactionResult(
            original_count=original_count,
            compacted_count=compacted_count,
            space_saved=space_saved,
            compacted_segments=compacted_ids,
            summary=f"Merged {original_count} -> {compacted_count} episodic memories ({space_saved:.1f}% saved)"
        )
    
    def _generate_summary(
        self,
        segments: List[MemorySegment],
        llm_summarizer: Any
    ) -> str:
        """Generate a summary of multiple segments using an LLM."""
        # Would call LLM to summarize
        combined = "\n".join([s.content for s in segments[:10]])  # Limit to avoid context overflow
        return f"Summary of {len(segments)} memory segments: {combined[:200]}..."
    
    def _compute_similarity(
        self,
        mem1: Dict[str, Any],
        mem2: Dict[str, Any]
    ) -> float:
        """Compute similarity between two memories."""
        # Simple heuristic - would use embeddings in real implementation
        obs1 = str(mem1.get("observation", ""))
        obs2 = str(mem2.get("observation", ""))
        
        # Jaccard similarity on words
        words1 = set(obs1.lower().split())
        words2 = set(obs2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _merge_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple memories into one."""
        if len(memories) == 1:
            return memories[0]
        
        # Keep the most recent memory as base
        sorted_memories = sorted(
            memories,
            key=lambda m: m.get("timestamp", ""),
            reverse=True
        )
        
        base = sorted_memories[0].copy()
        base["merged_from"] = [m.get("id", "unknown") for m in memories]
        base["merge_count"] = len(memories)
        
        return base
    
    def prune_stale_memories(
        self,
        memories: List[MemorySegment],
        max_age_days: int = 30
    ) -> List[MemorySegment]:
        """Remove memories older than max_age_days."""
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=max_age_days)
        
        return [m for m in memories if m.last_accessed >= cutoff]
