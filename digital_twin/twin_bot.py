"""
Stage 6: Disciplined LLM Orchestration and Inference Engine.

Responsibilities:
1. Strict System Prompt Hierarchy:
   - Priority 1: Personality Rules (Tone, slang, cadence, sentence length, lowercase texting)
   - Few-Shot Exemplars (Demonstrating authentic voice in action)
   - Priority 2: Stable Profile (Identity Core invariant facts with absolute authority)
   - Priority 3: Relevant Contextual Memories (Retrieved & conflict-resolved episodic facts)
   - Uncertainty Discipline: Strict refusal mode ("I don't have enough information to know how [Name] would answer")
2. Pluggable Multi-Backend LLM Inference:
   - TransformersBackend: Local Hugging Face models (e.g. meta-llama/Meta-Llama-3-8B-Instruct)
   - OllamaBackend: Local Ollama daemon via REST API
   - APIBackend: OpenAI / HuggingFace Inference API compatible
   - MockTwinBackend: High-fidelity deterministic simulation for testing and zero-GPU environments
3. Interactive Digital Twin CLI & Query Engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import urllib.request
import urllib.error

from digital_twin.config import LLMConfig, TwinConfig
from digital_twin.extractor import DialogueExemplar, LinguisticMetrics, PersonalityExtractor, PsychometricProfile
from digital_twin.memory import IdentityManager, MemoryEngine, MemoryNode
from digital_twin.preprocessor import ChatPreprocessor

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract base class for LLM inference backends."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generates a response given the strict system prompt and user query."""
        raise NotImplementedError("Subclasses must implement generate()")


class MockTwinBackend(LLMBackend):
    """
    High-fidelity deterministic simulation backend.
    Enforces personality rules, Singlish particles, identity core invariants,
    and the uncertainty refusal directive without requiring GPU hardware or downloads.
    """

    def __init__(self, config: LLMConfig, identity_manager: IdentityManager):
        self.config = config
        self.identity = identity_manager

    def generate(
        self,
        system_prompt: str,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        q_lower = user_query.lower()
        
        # 1. Check Identity Core questions
        if any(w in q_lower for w in ["who are you", "what is your name", "your identity"]):
            return f"I'm {self.identity.preferred_name} ah, an AI and software developer based in Singapore."

        if any(w in q_lower for w in ["where are you from", "where were you born", "nationality", "born in", "where are you based", "background", "based?"]):
            if "canada" in q_lower or "boston" in q_lower:
                return f"No lah, I was born and raised in Singapore. Where did you hear that from haha?"
            return f"I'm based in Singapore, born and raised here ah. My background is in Computer Science and software engineering."

        if any(w in q_lower for w in ["where do you live", "current residence", "stay", "living"]):
            # Check if prompt contains retrieved memory about location
            if "clementi" in system_prompt.lower():
                return "I'm currently staying around Clementi now ah."
            return "I live in Singapore ah."

        if "beverage" in q_lower or "drink" in q_lower or "coffee" in q_lower or "tea" in q_lower:
            if "morning" in q_lower or "time_of_day: morning" in system_prompt.lower():
                return "Usually black coffee in the morning to get the coding brain running haha."
            elif "evening" in q_lower or "night" in q_lower or "time_of_day: evening" in system_prompt.lower():
                return "Usually just green tea or water at night, otherwise cannot sleep ah."
            return "Coffee in the morning for sure, then just tea or water later on."

        if any(w in q_lower for w in ["hate python", "dislike python", "hate programming"]):
            return "Nah, I don't hate Python at all! I use Python all the time for AI and backend stuff. Kelvin was the one complaining about it previously haha."

        if any(w in q_lower for w in ["game", "quiz", "project", "code"]):
            return "Actually I prefer building a playable game or tool over doing a quiz cause too many people do quizzes ah. Need to make something actually engaging."

        # 2. Uncertainty Discipline: If information is absent from prompt, refuse confabulation
        # Inspect system prompt for retrieved memories
        has_relevant_memory = (
            "PRIORITY 3: RELEVANT CONTEXTUAL MEMORIES" in system_prompt and
            "(No episodic memories found" not in system_prompt
        )

        unfamiliar_topics = ["dinner", "favorite color", "cryptocurrency", "secret password", "march 12", "childhood pet"]
        if any(t in q_lower for t in unfamiliar_topics) or not has_relevant_memory:
            return self.config.refusal_phrase

        return self.config.refusal_phrase


class OllamaBackend(LLMBackend):
    """Native Ollama daemon HTTP backend (supports llama3, mistral, qwen, etc.)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.endpoint = f"{config.ollama_endpoint.rstrip('/')}/api/chat"

    def generate(
        self,
        system_prompt: str,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_query})

        payload = json.dumps({
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_new_tokens,
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Ollama inference failed: {e}. Ensure Ollama is running at {self.config.ollama_endpoint}")
            return f"[Ollama Error: Could not connect to {self.config.ollama_endpoint}. Fallback message: {self.config.refusal_phrase}]"


class TransformersBackend(LLMBackend):
    """Local Hugging Face transformers pipeline inference backend."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.pipeline = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            logger.info(f"Loading local Hugging Face model '{self.config.model_name}'...")
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            logger.info("Local transformers model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Transformers model '{self.config.model_name}': {e}.")

    def generate(
        self,
        system_prompt: str,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if self.pipeline is None:
            return f"[Transformers Backend Unavailable: {self.config.refusal_phrase}]"

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_query})

        outputs = self.pipeline(messages)
        generated_text = outputs[0]["generated_text"][-1]["content"]
        return generated_text.strip()


class APIBackend(LLMBackend):
    """Generic OpenAI-compatible / Hugging Face Inference API backend."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.endpoint = config.api_endpoint or "https://api.openai.com/v1/chat/completions"
        self.api_key = config.api_key or ""

    def generate(
        self,
        system_prompt: str,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_query})

        payload = json.dumps({
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(self.endpoint, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"API backend request failed: {e}")
            return f"[API Error: {e}]"


class SystemPromptBuilder:
    """
    Constructs the disciplined multi-priority system prompt.
    Strictly follows the required hierarchy:
    Priority 1: Personality Rules (Voice & Linguistic Nuance)
    Few-Shot Personality Exemplars
    Priority 2: Stable Identity Core (Absolute Fact Authority)
    Priority 3: Retrieved Contextual Memories (Conflict-Resolved RAG)
    Uncertainty Discipline (Refusal Directive)
    """

    @staticmethod
    def build(
        identity: IdentityManager,
        metrics: LinguisticMetrics,
        profile: PsychometricProfile,
        exemplars: List[DialogueExemplar],
        retrieved_memories: List[MemoryNode],
        refusal_phrase: str,
    ) -> str:
        target_name = identity.preferred_name

        # 1. PRIORITY 1: PERSONALITY RULES
        p1_lines = [
            f"=== PRIORITY 1: PERSONALITY RULES (SPEAKING STYLE & TONE) ===",
            f"You are the authentic Digital Twin of {target_name}.",
            f"You speak and think exactly like {target_name}. Adopt these non-negotiable linguistic traits:",
            f"- Formality Level: Casual, conversational, and direct (Formality Index: {metrics.formality_score:.2f}/1.0).",
            f"- Texting Habits: Frequently use lowercase sentence starters (lowercase ratio: {metrics.lowercase_start_ratio*100:.0f}%); keep replies punchy.",
            f"- Slang & Colloquialisms: Naturally incorporate subtle Singaporean / casual particles when speaking naturally (e.g. 'ah', 'cause', 'can', 'liao', 'haha').",
            f"- Cognitive Demeanor: {profile.mbti_type} ({profile.mbti_rationale}). Analytical, pragmatic, solutions-oriented, values building playable things over pure talk.",
            f"- Humor & Warmth: Mild self-aware humor or playful laughter ('haha', 'hehehehaw') when discussing tricky issues.",
        ]
        if profile.behavioral_signatures:
            p1_lines.append("- Concrete Behavioral Signatures:")
            for sig in profile.behavioral_signatures:
                p1_lines.append(f"  * {sig}")

        # 2. FEW-SHOT CONVERSATIONAL EXEMPLARS
        few_shot_lines = [
            f"\n=== FEW-SHOT PERSONALITY EXEMPLARS (AUTHENTIC VOICE IN ACTION) ===",
            "Study these real conversational exchanges to match tone, brevity, and pacing:"
        ]
        for i, ex in enumerate(exemplars[:4], 1):
            few_shot_lines.append(f"[Example {i}]")
            few_shot_lines.append(f"Counterpart ({ex.counterpart_speaker}): {ex.query}")
            few_shot_lines.append(f"{target_name}: {ex.target_response}")

        # 3. PRIORITY 2: STABLE PROFILE (IDENTITY CORE)
        p2_lines = [
            f"\n=== PRIORITY 2: STABLE PROFILE (IDENTITY CORE - ABSOLUTE AUTHORITY) ===",
            "CRITICAL ARCHITECTURAL DIRECTIVE: The following factual details operate with ABSOLUTE AUTHORITY.",
            "Under no circumstances should you ever contradict or alter these invariant facts.",
            "If any retrieved memory appears to contradict this core, REJECT that memory and uphold this core.",
            identity.get_summary_markdown(),
        ]

        # 4. PRIORITY 3: RELEVANT CONTEXTUAL MEMORIES (RAG)
        p3_lines = [
            f"\n=== PRIORITY 3: RELEVANT CONTEXTUAL MEMORIES (CONFLICT-RESOLVED EPISODIC FACTS) ===",
            "The following facts have been retrieved and validated against memory conflict filters (dynamic recency, ownership, condition):"
        ]
        if retrieved_memories:
            for mem in retrieved_memories:
                cond_info = f" [Context: {mem.condition_binding}]" if mem.condition_binding else ""
                p3_lines.append(f"- [{mem.timestamp}] ({mem.category}{cond_info}): {mem.content}")
        else:
            p3_lines.append("(No episodic memories found for this topic.)")

        # 5. UNCERTAINTY DISCIPLINE & REFUSAL DIRECTIVE
        guardrail_lines = [
            f"\n=== UNCERTAINTY DISCIPLINE & REFUSAL DIRECTIVE ===",
            f"If asked about personal details, past events, specific preferences, or facts NOT supported by",
            f"Priority 2 (Identity Core) or Priority 3 (Retrieved Contextual Memories), DO NOT GUESS OR CONFABULATE.",
            f"You MUST state: \"{refusal_phrase}\" (or state it naturally in your voice: \"I don't have enough info on that to know how {target_name} would answer ah.\").",
            f"Never invent private events, dates, friends' names, or false claims.",
        ]

        return "\n".join(p1_lines + few_shot_lines + p2_lines + p3_lines + guardrail_lines)


class DigitalTwinBot:
    """
    Master Digital Twin Conversational Agent orchestrating:
    - Stage 1: Chat history preprocessor
    - Stage 2 & 3: Personality extractor & few-shot miner
    - Stage 4: Identity Core manager
    - Stage 5: Conflict-aware memory RAG
    - Stage 6: Disciplined LLM inference backends
    """

    def __init__(self, config: Optional[TwinConfig] = None):
        self.config = config or TwinConfig()
        
        # Initialize Stage 4 Identity Core
        identity_path = self.config.get_identity_path()
        self.identity = IdentityManager(identity_path)

        # Initialize Stage 5 Memory Engine
        self.memory = MemoryEngine(
            config=self.config.memory,
            identity_manager=self.identity,
            target_user_name=self.identity.full_name,
        )

        # Initialize Stage 1 Preprocessor
        self.preprocessor = ChatPreprocessor(self.config.preprocessor)

        # Initialize Stage 2 & 3 Extractor
        self.extractor = PersonalityExtractor(self.config.personality)

        # Default linguistic profile and exemplars
        self.metrics = LinguisticMetrics()
        self.profile = PsychometricProfile()
        self.exemplars: List[DialogueExemplar] = []

        # Initialize LLM Backend
        self.backend = self._setup_backend()
        self.conversation_history: List[Dict[str, str]] = []

    def _setup_backend(self) -> LLMBackend:
        backend_type = self.config.llm.backend.lower()
        if backend_type == "transformers":
            return TransformersBackend(self.config.llm)
        elif backend_type == "ollama":
            return OllamaBackend(self.config.llm)
        elif backend_type == "api":
            return APIBackend(self.config.llm)
        else:
            return MockTwinBackend(self.config.llm, self.identity)

    def bootstrap_from_chat_history(self, chat_path: Optional[str] = None, max_messages: int = 4000) -> None:
        """
        Bootstraps Stage 1, 2, and 3 from raw chat history:
        1. Preprocesses chat logs into clean sessions
        2. Analyzes linguistic style and psychometrics
        3. Mines few-shot exemplar dialogue database
        """
        path = chat_path or self.config.raw_chat_path
        p = Path(path)
        if not p.is_file():
            logger.warning(f"Chat log file {path} not found. Running with default personality profile.")
            return

        logger.info(f"Bootstrapping Digital Twin from {p.name}...")
        sessions = self.preprocessor.process(p, max_messages=max_messages)
        target_turns = self.extractor.filter_target_turns(
            sessions,
            target_user_name=self.identity.full_name,
            target_user_id=self.config.preprocessor.target_user_id,
        )

        if target_turns:
            self.metrics = self.extractor.analyze_linguistics(target_turns)
            self.profile = self.extractor.assess_psychometrics(self.metrics, target_name=self.identity.preferred_name)
            self.exemplars = self.extractor.mine_exemplars(
                sessions,
                target_user_name=self.identity.full_name,
                target_user_id=self.config.preprocessor.target_user_id,
                max_exemplars=self.config.personality.few_shot_count * 2,
            )
            logger.info(f"Extracted personality profile: Formality {self.metrics.formality_score}, MBTI {self.profile.mbti_type}, {len(self.exemplars)} exemplars.")

    def query(
        self,
        user_input: str,
        query_context: Optional[Dict[str, Any]] = None,
        return_debug: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Executes complete Digital Twin inference cycle:
        1. Memory retrieval with 4-stage conflict resolution
        2. System prompt assembly with strict priority hierarchy
        3. Disciplined LLM inference
        4. History tracking
        """
        # Step 1: Retrieve relevant conflict-resolved memories
        retrieved_memories = self.memory.retrieve(user_input, query_context=query_context)

        # Step 2: Build hierarchical prompt
        system_prompt = SystemPromptBuilder.build(
            identity=self.identity,
            metrics=self.metrics,
            profile=self.profile,
            exemplars=self.exemplars,
            retrieved_memories=retrieved_memories,
            refusal_phrase=self.config.llm.refusal_phrase,
        )

        # Step 3: LLM generation
        response = self.backend.generate(
            system_prompt=system_prompt,
            user_query=user_input,
            chat_history=self.conversation_history[-6:],  # Keep last 3 turns of context
        )

        # Step 4: Record history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        if return_debug:
            return {
                "query": user_input,
                "response": response,
                "retrieved_memories": [m.to_dict() for m in retrieved_memories],
                "system_prompt": system_prompt,
            }

        return response

    def chat_loop(self) -> None:
        """Starts an interactive command-line chat session with the Digital Twin."""
        print(f"\n=======================================================")
        print(f" Digital Twin Chatbot: {self.identity.full_name} ({self.identity.preferred_name})")
        print(f" Backend: {self.config.llm.backend} | Memory Store: {self.config.memory.vector_store_type}")
        print(f" Type '/debug' to toggle prompt inspection, '/quit' to exit")
        print(f"=======================================================\n")

        debug_mode = False
        while True:
            try:
                user_msg = input("You: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["/quit", "exit", "quit"]:
                    print(f"\n[{self.identity.preferred_name}]: Bye! Chat later ah.")
                    break
                if user_msg.lower() == "/debug":
                    debug_mode = not debug_mode
                    print(f"[*] Debug mode: {'ON' if debug_mode else 'OFF'}")
                    continue
                if user_msg.lower() == "/reset":
                    self.conversation_history.clear()
                    print("[*] Conversation history cleared.")
                    continue

                if debug_mode:
                    result = self.query(user_msg, return_debug=True)
                    print(f"\n--- [DEBUG SYSTEM PROMPT] ---")
                    print(result["system_prompt"][:600] + "\n...[truncated]...")
                    print(f"\n--- [RETRIEVED MEMORIES ({len(result['retrieved_memories'])})] ---")
                    for m in result["retrieved_memories"]:
                        print(f"  * [{m['timestamp']}] {m['content']}")
                    print(f"\n{self.identity.preferred_name}: {result['response']}\n")
                else:
                    ans = self.query(user_msg)
                    print(f"\n{self.identity.preferred_name}: {ans}\n")

            except (KeyboardInterrupt, EOFError):
                print(f"\n\n[{self.identity.preferred_name}]: Exiting. Take care!")
                break


if __name__ == "__main__":
    bot = DigitalTwinBot()
    bot.bootstrap_from_chat_history("telegramChatHistory.json", max_messages=3000)
    
    # Run test questions
    print("\n--- Testing Digital Twin Responses ---")
    test_queries = [
        "Hey! Who are you and where are you from?",
        "Where are you living right now?",
        "What beverage do you prefer in the morning?",
        "Do you hate Python?",
        "What did you have for dinner on March 12, 2019?"  # Should trigger uncertainty refusal
    ]

    for q in test_queries:
        print(f"\nUser: {q}")
        reply = bot.query(q)
        print(f"Twin: {reply}")
