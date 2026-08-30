"""
Citation Manager for tracking sources and generating markdown footnotes.

Provides a centralized citation management system that tracks all sources
referenced during research and generates proper markdown footnote citations.

Sources are grouped by *research topic* — each rabbit hole / sub-question
gets its own bucket so the bibliography stays organized by ask rather than
dumping every source into one flat list. The AI sets the active topic before
registering sources for a new line of inquiry; sources then inherit that
topic automatically.

Credibility weighting is NOT hardcoded. Sources default to 0.0 (unassessed)
until the AI explicitly calls ``assess_source`` with a 0–1 evidence strength
and rationale. ``assess_source`` applies a hard safety cap per source type
(academic 1.0, docs 0.95, news 0.85, web 0.80, forum 0.65, social/other 0.60)
— the cap bounds but never assigns a flat rating.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


DEFAULT_TOPIC = "general"


class SourceType(Enum):
    """Types of sources that can be cited."""
    WEB = "web"
    ACADEMIC = "academic"
    SOCIAL = "social"
    FORUM = "forum"
    NEWS = "news"
    DOCUMENTATION = "documentation"
    OTHER = "other"


# Hard safety caps per source type — these BOUND the AI's assessment,
# they do not assign a flat rating. A weak academic paper can score 0.2;
# an excellent web source can be strong but cannot exceed 0.80.
SOURCE_TYPE_CAPS: Dict[SourceType, float] = {
    SourceType.ACADEMIC: 1.0,
    SourceType.DOCUMENTATION: 0.95,
    SourceType.NEWS: 0.85,
    SourceType.WEB: 0.80,
    SourceType.FORUM: 0.65,
    SourceType.SOCIAL: 0.60,
    SourceType.OTHER: 0.60,
}


@dataclass
class Source:
    """Represents a citable source."""
    id: int
    title: str
    url: str
    source_type: SourceType
    authors: List[str] = field(default_factory=list)
    date: Optional[str] = None
    accessed_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    credibility_score: float = 0.0
    topic: str = DEFAULT_TOPIC

    def __post_init__(self):
        """Validate source data after initialization."""
        if not self.title:
            raise ValueError("Source title is required")
        if not self.url:
            raise ValueError("Source URL is required")


def source_type_cap(source_type: SourceType) -> float:
    """Return the hard safety cap for a source type (0.0–1.0)."""
    if isinstance(source_type, str):
        try:
            source_type = SourceType(source_type.lower())
        except ValueError:
            source_type = SourceType.OTHER
    elif not isinstance(source_type, SourceType):
        source_type = SourceType.OTHER
    return SOURCE_TYPE_CAPS.get(source_type, 0.60)


def calculate_credibility_score(source_type: SourceType, metadata: Dict[str, Any]) -> float:
    """Deprecated stub — credibility is now AI-assessed via ``assess_source``.

    Kept for backward-compatibility imports. Always returns 0.0 (unassessed).
    The old hardcoded base-score + boost table was removed so the AI owns
    the weighting decision when it adds a source to the bibliography.
    """
    return 0.0


class CitationManager:
    """
    Manages citations for research sources, grouped by topic.

    Tracks all sources referenced during research and generates markdown
    footnote citations in the format [^n] where n is the citation number.
    Sources are bucketed by topic so the bibliography can be compiled
    per-ask rather than as one undifferentiated dump.

    Example:
        >>> manager = CitationManager()
        >>> manager.set_topic("schema world models")
        >>> manager.add_source(
        ...     title="Python Documentation",
        ...     url="https://docs.python.org/3/",
        ...     source_type=SourceType.DOCUMENTATION
        ... )
        >>> manager.generate_citation(1)
        '[^1]'
        >>> manager.compile_bibliography()
        '## Bibliography\\n\\n### schema world models\\n\\n[^1]: Python Documentation. https://docs.python.org/3/. Accessed: 2024-01-15'
    """

    def __init__(self):
        """Initialize the citation manager."""
        self._sources: Dict[int, Source] = {}
        self._next_id: int = 1
        self._source_urls: Dict[str, int] = {}  # URL to ID mapping for deduplication
        self._current_topic: str = DEFAULT_TOPIC
        self._topic_order: List[str] = []  # topics in first-seen order

    # ---------------------------------------------------------------- topic

    def set_topic(self, topic: str) -> str:
        """Set the active research topic.

        Sources registered afterwards inherit this topic until it is changed
        again. Use this when pivoting to a new rabbit hole so the
        bibliography stays grouped by ask. An empty/whitespace topic falls
        back to the default bucket.
        """
        topic = (topic or "").strip() or DEFAULT_TOPIC
        if topic != self._current_topic and topic not in self._topic_order:
            self._topic_order.append(topic)
        self._current_topic = topic
        return self._current_topic

    def get_current_topic(self) -> str:
        """Return the currently active research topic."""
        return self._current_topic

    def list_topics(self) -> List[str]:
        """Return all topics that have at least one source, in first-seen order."""
        seen: List[str] = []
        for cid in sorted(self._sources.keys()):
            topic = self._sources[cid].topic
            if topic not in seen:
                seen.append(topic)
        return seen

    # -------------------------------------------------------------- sources

    def add_source(
        self,
        title: str,
        url: str,
        source_type: SourceType,
        authors: Optional[List[str]] = None,
        date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
    ) -> int:
        """
        Register a new source and return its citation ID.

        If a source with the same URL already exists, returns the existing ID
        instead of creating a duplicate. The source is tagged with the active
        topic (or an explicit ``topic`` override) so the bibliography can be
        compiled per-ask.

        Credibility defaults to 0.0 (unassessed). Call ``assess_source`` after
        evaluating the source to assign an AI-driven evidence weight.

        Args:
            title: The title of the source
            url: The URL of the source
            source_type: The type of source (web, academic, social, etc.)
            authors: Optional list of authors
            date: Optional publication date
            metadata: Optional additional metadata (e.g., score, votes, etc.)
            topic: Optional topic override; defaults to the active topic

        Returns:
            The citation ID for this source

        Example:
            >>> manager.add_source(
            ...     title="Example Article",
            ...     url="https://example.com/article",
            ...     source_type=SourceType.WEB,
            ...     authors=["John Doe", "Jane Smith"],
            ...     date="2024-01-15"
            ... )
            1
        """
        # Check for duplicate URLs
        if url in self._source_urls:
            return self._source_urls[url]

        # Coerce string → enum at the boundary. The type hint says
        # `SourceType` but several call sites in mu/tools/research/handlers.py pass
        # plain strings ("web", "academic", ...) — those used to land
        # in storage verbatim and crash any consumer that did
        # `source.source_type.value`. Normalize here so every Source
        # in storage carries a proper enum, regardless of caller.
        if isinstance(source_type, str):
            try:
                source_type = SourceType(source_type.lower())
            except ValueError:
                source_type = SourceType.OTHER
        elif not isinstance(source_type, SourceType):
            source_type = SourceType.OTHER

        resolved_topic = (topic or "").strip() or self._current_topic
        if resolved_topic not in self._topic_order:
            self._topic_order.append(resolved_topic)

        citation_id = self._next_id

        source = Source(
            id=citation_id,
            title=title,
            url=url,
            source_type=source_type,
            authors=authors or [],
            date=date,
            metadata=metadata or {},
            credibility_score=0.0,  # unassessed until assess_source is called
            topic=resolved_topic,
        )

        self._sources[citation_id] = source
        self._source_urls[url] = citation_id
        self._next_id += 1

        return citation_id

    def get_source(self, citation_id: int) -> Optional[Source]:
        """
        Get a source by its citation ID.

        Args:
            citation_id: The citation ID to look up

        Returns:
            The Source object if found, None otherwise
        """
        return self._sources.get(citation_id)

    def assess_source(self, citation_id: int, importance: float, rationale: str = "") -> Source:
        """Apply the model's evidence assessment, bounded by safety caps.

        The AI owns the weighting: ``importance`` is a 0–1 evidence-strength
        rating based on authority, methodology, relevance, recency, and
        corroboration. The source type's hard cap (see ``SOURCE_TYPE_CAPS``)
        bounds but never flatly assigns the score.
        """
        source = self.get_source(citation_id)
        if source is None:
            raise ValueError(f"Citation ID {citation_id} not found")
        requested = max(0.0, min(1.0, float(importance)))
        cap = source_type_cap(source.source_type)
        source.credibility_score = min(requested, cap)
        source.metadata["model_importance"] = requested
        source.metadata["credibility_cap"] = cap
        if rationale.strip():
            source.metadata["assessment_rationale"] = rationale.strip()[:1000]
        return source

    def generate_citation(self, citation_id: int) -> str:
        """
        Generate a markdown footnote citation reference.

        Args:
            citation_id: The citation ID to reference

        Returns:
            A markdown footnote reference in the format [^n]

        Raises:
            ValueError: If the citation_id does not exist

        Example:
            >>> manager.generate_citation(1)
            '[^1]'
        """
        if citation_id not in self._sources:
            raise ValueError(f"Citation ID {citation_id} not found")

        return f"[^{citation_id}]"

    def _format_authors(self, authors: List[str]) -> str:
        """Format a list of authors for citation."""
        if not authors:
            return ""

        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        else:
            return f"{authors[0]} et al."

    def _format_bibliography_entry(self, source: Source) -> str:
        """
        Format a single bibliography entry with credibility indicators.

        Args:
            source: The source to format

        Returns:
            A formatted bibliography entry string
        """
        parts = [f"[^{source.id}]: {source.title}."]

        # Add authors if present
        if source.authors:
            author_str = self._format_authors(source.authors)
            parts[0] = f"[^{source.id}]: {author_str}. {source.title}."

        # Add URL
        parts.append(source.url)

        # Add date if present
        if source.date:
            parts.append(f"Published: {source.date}.")

        # Add accessed date
        parts.append(f"Accessed: {source.accessed_date}.")

        # Add source type indicator
        type_indicator = {
            SourceType.ACADEMIC: "[Academic]",
            SourceType.SOCIAL: "[Social]",
            SourceType.FORUM: "[Forum]",
            SourceType.NEWS: "[News]",
            SourceType.DOCUMENTATION: "[Documentation]",
            SourceType.WEB: "[Web]",
            SourceType.OTHER: ""
        }

        if type_indicator.get(source.source_type):
            parts.append(type_indicator[source.source_type])

        # Add credibility score with stars (only if assessed)
        if source.credibility_score > 0:
            stars = "★" * int(round(source.credibility_score * 5))
            stars_empty = "☆" * (5 - int(round(source.credibility_score * 5)))
            parts.append(f"(Credibility: {stars}{stars_empty} {source.credibility_score:.1f}/1.0)")
        else:
            parts.append("(Credibility: unassessed)")

        return " ".join(parts)

    def _sources_for_topic(self, topic: Optional[str]) -> List[Source]:
        """Return sources for a topic (or all if topic is None), sorted by id."""
        if topic is None:
            return self.get_all_sources()
        return [
            self._sources[cid]
            for cid in sorted(self._sources.keys())
            if self._sources[cid].topic == topic
        ]

    def _ordered_topics(self, topic: Optional[str]) -> List[str]:
        """Return topics in first-seen order, optionally filtered to one."""
        if topic is not None:
            return [topic] if any(
                s.topic == topic for s in self._sources.values()
            ) else []
        # Preserve first-seen order across stored sources.
        ordered: List[str] = []
        for cid in sorted(self._sources.keys()):
            t = self._sources[cid].topic
            if t not in ordered:
                ordered.append(t)
        return ordered

    def compile_bibliography(self, topic: Optional[str] = None) -> str:
        """
        Compile cited sources into a bibliography, grouped by research topic.

        When ``topic`` is None, every topic bucket is emitted in first-seen
        order, each under its own ``### <topic>`` heading. When ``topic`` is
        given, only that bucket is emitted (still under its heading).

        Args:
            topic: Optional topic filter; None emits all topics grouped.

        Returns:
            A formatted bibliography section with sources grouped by topic

        Example:
            >>> manager.compile_bibliography()
            '## Bibliography\\n\\n### general\\n\\n[^1]: ...'
        """
        if not self._sources:
            return ""

        lines = ["## Bibliography", ""]

        for t in self._ordered_topics(topic):
            sources = self._sources_for_topic(t)
            if not sources:
                continue
            lines.append(f"### {t}")
            lines.append("")
            for source in sources:
                lines.append(self._format_bibliography_entry(source))
                lines.append("")

        return "\n".join(lines).strip()

    def clear(self) -> None:
        """Clear all stored sources and reset the citation counter."""
        self._sources.clear()
        self._source_urls.clear()
        self._next_id = 1
        self._current_topic = DEFAULT_TOPIC
        self._topic_order = []

    @property
    def source_count(self) -> int:
        """Return the number of registered sources."""
        return len(self._sources)

    def get_all_sources(self) -> List[Source]:
        """Return all registered sources in citation order."""
        return [self._sources[cid] for cid in sorted(self._sources.keys())]

    def get_sources_by_topic(self, topic: str) -> List[Source]:
        """Return sources tagged with ``topic``, in citation order."""
        return [
            self._sources[cid]
            for cid in sorted(self._sources.keys())
            if self._sources[cid].topic == topic
        ]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return a JSON-safe snapshot suitable for session persistence."""
        return [
            {
                "id": source.id,
                "title": source.title,
                "url": source.url,
                "source_type": source.source_type.value,
                "authors": list(source.authors or []),
                "date": source.date,
                "accessed_date": source.accessed_date,
                "metadata": dict(source.metadata or {}),
                "credibility_score": source.credibility_score,
                "topic": source.topic,
            }
            for source in self.get_all_sources()
        ]

    def load_dict(self, records: Any) -> None:
        """Replace sources from a persisted session snapshot.

        Source IDs are retained so citations already present in the saved
        conversation continue to point at the same bibliography entries.
        Invalid legacy records are ignored rather than preventing a session
        from loading. Legacy records without a ``topic`` field fall back to
        the default bucket so older sessions still load cleanly.
        """
        self.clear()
        if not isinstance(records, list):
            return
        for raw in records:
            if not isinstance(raw, dict):
                continue
            try:
                source_id = int(raw.get("id"))
                title = str(raw.get("title") or "").strip()
                url = str(raw.get("url") or "").strip()
                if source_id < 1 or not title or not url:
                    continue
                source_type = SourceType(str(raw.get("source_type") or "other"))
            except (TypeError, ValueError):
                continue
            topic = str(raw.get("topic") or "").strip() or DEFAULT_TOPIC
            if topic not in self._topic_order:
                self._topic_order.append(topic)
            source = Source(
                id=source_id,
                title=title,
                url=url,
                source_type=source_type,
                authors=list(raw.get("authors") or []),
                date=raw.get("date"),
                accessed_date=str(raw.get("accessed_date") or datetime.now().strftime("%Y-%m-%d")),
                metadata=dict(raw.get("metadata") or {}),
                credibility_score=float(raw.get("credibility_score") or 0.0),
                topic=topic,
            )
            # Preserve URL de-duplication and the original citation number.
            if url in self._source_urls or source_id in self._sources:
                continue
            self._sources[source_id] = source
            self._source_urls[url] = source_id
            self._next_id = max(self._next_id, source_id + 1)
        # Restore the last-seen topic as active, if any.
        if self._topic_order:
            self._current_topic = self._topic_order[-1]


# Global citation manager instance for use across tools
_citation_manager: Optional[CitationManager] = None


def get_citation_manager() -> CitationManager:
    """
    Get the global citation manager instance.

    Creates a new instance if one doesn't exist.

    Returns:
        The global CitationManager instance
    """
    global _citation_manager
    if _citation_manager is None:
        _citation_manager = CitationManager()
    return _citation_manager


def reset_citation_manager() -> None:
    """Reset the global citation manager to a fresh instance."""
    global _citation_manager
    _citation_manager = CitationManager()


def set_research_topic(topic: str) -> str:
    """Set the active research topic on the global citation manager.

    Call this when pivoting to a new rabbit hole so sources registered
    afterwards are grouped under the right ask in the bibliography.
    """
    return get_citation_manager().set_topic(topic)


def get_current_research_topic() -> str:
    """Return the active research topic on the global citation manager."""
    return get_citation_manager().get_current_topic()


def register_source(
    title: str,
    url: str,
    source_type: SourceType,
    authors: Optional[List[str]] = None,
    date: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None,
) -> int:
    """
    Convenience function to register a source with the global citation manager.

    Sources registered through this research-tools entry point carry a
    conservative type-based prior credibility (half the type's hard cap) so
    the /research status surface reflects provisional evidence quality
    before an explicit assess_source() call. Unassessed sources added via
    CitationManager.add_source directly remain 0.0.

    Args:
        title: The title of the source
        url: The url of the source
        source_type: The type of source
        authors: Optional list of authors
        date: Optional publication date
        metadata: Optional additional metadata
        topic: Optional topic override; defaults to the active topic

    Returns:
        The citation ID for this source
    """
    citation_id = get_citation_manager().add_source(
        title=title,
        url=url,
        source_type=source_type,
        authors=authors,
        date=date,
        metadata=metadata,
        topic=topic,
    )
    source = get_citation_manager().get_source(citation_id)
    if source is not None and source.credibility_score == 0.0:
        # Conservative prior: half the type's hard cap. assess_source()
        # overrides this with the model's evidence assessment.
        source.credibility_score = round(source_type_cap(source_type) / 2.0, 3)
    return citation_id


def get_citation(citation_id: int) -> str:
    """
    Convenience function to get a citation reference.

    Args:
        citation_id: The citation ID

    Returns:
        The markdown footnote reference
    """
    return get_citation_manager().generate_citation(citation_id)


def compile_bibliography(topic: Optional[str] = None) -> str:
    """
    Convenience function to compile the bibliography, grouped by topic.

    Args:
        topic: Optional topic filter; None emits all topics grouped.

    Returns:
        The formatted bibliography section
    """
    return get_citation_manager().compile_bibliography(topic=topic)