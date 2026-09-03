"""
Digital Twin conversational architecture package.
Implements the 6-stage pipeline:
1. Chat History Preprocessing (Burst concatenation, session segmentation, PII cleaning)
2. Personality Extraction Engine (Linguistic style & psychometric Big-Five/MBTI profiling)
3. Personality Dataset & Few-Shot Miner (Exemplar mining and ChatML/LoRA formatting)
4. Stable Information Engine (Identity Core with strict fact hierarchy)
5. Memory Engine & Conflict-Aware RAG (Dynamic, static, conditional conflict resolution)
6. Disciplined LLM Orchestration & Inference (Priority prompt hierarchy & uncertainty discipline)
"""

from digital_twin.config import (
    PreprocessorConfig,
    PersonalityConfig,
    MemoryConfig,
    LLMConfig,
    TwinConfig,
)
from digital_twin.preprocessor import (
    ChatPreprocessor,
    ChatTurn,
    ChatSession,
    RawMessage,
)
from digital_twin.extractor import (
    PersonalityExtractor,
    LinguisticMetrics,
    PsychometricProfile,
    DialogueExemplar,
)
from digital_twin.memory import (
    IdentityManager,
    MemoryEngine,
    MemoryNode,
    SQLiteVectorStore,
    FastEmbedder,
)
from digital_twin.twin_bot import (
    DigitalTwinBot,
    SystemPromptBuilder,
    MockTwinBackend,
    OllamaBackend,
    TransformersBackend,
    APIBackend,
)

__all__ = [
    "PreprocessorConfig",
    "PersonalityConfig",
    "MemoryConfig",
    "LLMConfig",
    "TwinConfig",
    "ChatPreprocessor",
    "ChatTurn",
    "ChatSession",
    "RawMessage",
    "PersonalityExtractor",
    "LinguisticMetrics",
    "PsychometricProfile",
    "DialogueExemplar",
    "IdentityManager",
    "MemoryEngine",
    "MemoryNode",
    "SQLiteVectorStore",
    "FastEmbedder",
    "DigitalTwinBot",
    "SystemPromptBuilder",
    "MockTwinBackend",
    "OllamaBackend",
    "TransformersBackend",
    "APIBackend",
]
