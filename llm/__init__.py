"""
LLM Personality Modeling, Prompt Synthesis, SFT Dataset Compilation & Inference Module.

Provides linguistic metrics profiling, academic Big-5 personality scoring, dynamic system prompt
synthesis, ChatML conversational SFT dataset formatting, interactive CLI sandbox, and QLoRA fine-tuning.
"""

from llm.profiler import ContextProfile, MultiContextProfiler, build_default_profile
from llm.synthesizer import SystemPromptSynthesizer
from llm.export_training_data import DatasetExporter, format_conversation_for_sft
from llm.chat_app import PersonalityChatApp, RefusalDeflectionLayer, SimulatedClient

__all__ = [
    "ContextProfile",
    "MultiContextProfiler",
    "build_default_profile",
    "SystemPromptSynthesizer",
    "DatasetExporter",
    "format_conversation_for_sft",
    "PersonalityChatApp",
    "RefusalDeflectionLayer",
    "SimulatedClient",
]
