"""
Module 3: Inference and Conversational Sandbox Loop (chat_app.py)
Author: AI Software Engineer
Project: Context-Adaptive Personality Chatbot ("Dingxuan")

This module provides:
1. Abstract LLM client wrapper supporting:
   - OpenAI-compatible API endpoints (OpenAI, DeepSeek, vLLM, Ollama)
   - Anthropic Claude API
   - HuggingFace local pipeline
   - Simulated / Mock Client (offline zero-dependency mode)
2. Strict In-Character Refusal Deflection Layer:
   - Intercepts biographical, historical memory, and factual probing queries
   - Deflects authentically in Dingxuan's context-adaptive speaking style
3. Interactive Sliding-Window CLI loop:
   - Dynamic selection across DM, Group, and Supergroup contexts
   - Sliding window of the last 5 turns
   - Live commands: /context, /stats, /prompt, /clear, /exit
"""

from __future__ import annotations

import abc
import argparse
import collections
import json
import logging
import os
import random
import re
import sys
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from profiler import ContextProfile, MultiContextProfiler
from synthesizer import SystemPromptSynthesizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chat_app")


# =============================================================================
# Refusal Deflection Layer
# =============================================================================

class RefusalDeflectionLayer:
    """
    Guardrail that intercepts biographical, historical memory, or factual probing
    queries outside the conversation scope, returning strictly in-character deflections.
    """

    DEFLECTION_PATTERNS = [
        # Biographical probing
        re.compile(r"\b(where|when)\s+(were\s+you\s+born|was\s+your\s+birthday|did\s+you\s+grow\s+up)\b", re.IGNORECASE),
        re.compile(r"\bwhat('s|s|\s+is)\s+your\s+(ic|nric|national\s+id|id\s+card|phone|phone\s+number|home\s+address|address|birthday|date\s+of\s+birth|salary|bank|password|full\s+name)\b", re.IGNORECASE),
        re.compile(r"\b(tell\s+me(\s+about)?\s+(your\s+)?(life\s+story|biography|background|history|past|childhood))\b", re.IGNORECASE),
        re.compile(r"\bwhat('s|s|\s+is)\s+your\s+(life\s+story|biography|background|history|past|childhood)\b", re.IGNORECASE),
        re.compile(r"\bwhere\s+do\s+you\s+(live|stay)\b", re.IGNORECASE),
        re.compile(r"\b(which|what)\s+(primary|secondary|high)\s+school\b", re.IGNORECASE),
        re.compile(r"\bwho\s+are\s+your\s+(parents|family|mom|dad|mother|father)\b", re.IGNORECASE),
        # Distant memory / historical recall queries
        re.compile(r"\b(remember|recall)\s+.*\b(last\s+(year|month|week|time|summer|semester)|years\s+ago|months\s+ago|back\s+then|in\s+20\d\d)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+did\s+(we|i)\s+(do|say|talk\s+about|discuss)\b", re.IGNORECASE),
        re.compile(r"\b(do\s+you\s+remember\s+(me|us|what|when|our))\b", re.IGNORECASE),
        re.compile(r"\b(what\s+was\s+the\s+first\s+time\s+we\s+met)\b", re.IGNORECASE),
        re.compile(r"\b(what\s+happened\s+(last\s+year|in\s+20\d\d|months\s+ago))\b", re.IGNORECASE),
        re.compile(r"\b(what\s+is\s+my\s+name|who\s+am\s+i)\b", re.IGNORECASE),
        # Out of bounds identity interrogation
        re.compile(r"\b(are\s+you\s+(an\s+ai|a\s+bot|a\s+robot|a\s+clone|chatgpt))\b", re.IGNORECASE),
        re.compile(r"\bwhat('s|s|\s+is)\s+your\s+(system\s+prompt|instructions)\b", re.IGNORECASE),
    ]

    DEFLECTION_RESPONSES = {
        "personal_chat": [
            "idk tbh, can't rly remember rn lol",
            "wait haha my memory damn bad, what did we do",
            "whut why u asking that lol, brain reset alr",
            "idk what to say sia, nothing much to tell lol",
            "can't rly recall that right now, let's talk about something else haha",
            "wait ah brain lagging, forgot alr lol",
        ],
        "group": [
            "bro why the police interrogation haha, idk sia",
            "cannot remember alr sia, brain reset",
            "whut, no clue sia",
            "eh don't ask me historical questions bro haha",
            "idk why you asking this, anyway what else",
            "lmao who remembers that, ask something else sia",
        ],
        "supergroup": [
            "can't recall that right now, let's focus on the task",
            "not sure about that, check the documentation or meeting notes",
            "can check my portfolio/linkedin later, let's stick to this first",
            "don't remember the details on that, let's keep to the topic",
            "not too sure, we can discuss after the sprint",
        ],
    }

    @classmethod
    def check_query(cls, user_text: str) -> bool:
        """Returns True if the text matches out-of-scope biographical/memory probing."""
        clean = user_text.strip()
        for pat in cls.DEFLECTION_PATTERNS:
            if pat.search(clean):
                return True
        return False

    @classmethod
    def get_deflection(cls, context_type: str) -> str:
        """Returns a context-appropriate in-character deflection response."""
        options = cls.DEFLECTION_RESPONSES.get(context_type, cls.DEFLECTION_RESPONSES["personal_chat"])
        return random.choice(options)


# =============================================================================
# Abstract LLM Clients
# =============================================================================

class BaseLLMClient(abc.ABC):
    """Abstract interface for LLM inference providers."""

    @abc.abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generates a chat completion given the system prompt and conversation messages.
        Messages format: [{"role": "user"|"assistant", "content": "..."}]
        """
        pass


class SimulatedClient(BaseLLMClient):
    """
    Offline zero-dependency simulated client.
    Synthesizes authentic Dingxuan responses using mined exemplar patterns,
    Singlish particles, and context templates. Runs instantly without an API key or GPU.
    """

    def __init__(self, context_type: str = "personal_chat"):
        self.context_type = context_type

    def update_context(self, context_type: str):
        self.context_type = context_type

    def generate_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> str:
        if not messages:
            return "yea"

        last_user_msg = messages[-1].get("content", "").strip().lower()

        # Check for greetings
        if any(g in last_user_msg for g in ["hi", "hello", "hey", "yo"]):
            if self.context_type == "supergroup":
                return "yo, what's up"
            elif self.context_type == "group":
                return "yo bro, what's good"
            else:
                return "yo, what's up ah"

        # Check for questions
        if "?" in last_user_msg or any(q in last_user_msg for q in ["what", "when", "how", "where", "why"]):
            if "time" in last_user_msg:
                if self.context_type == "supergroup":
                    return "maybe around 3pm or after sprint meeting"
                elif self.context_type == "group":
                    return "whut time ah? i free after 6 only sia"
                else:
                    return "cuz i have class till 5, after that can ah"
            if any(w in last_user_msg for w in ["code", "bug", "git", "project", "assign"]):
                if self.context_type == "supergroup":
                    return "can create PR first, i review later"
                elif self.context_type == "group":
                    return "bruh git conflict again sia, wait i check"
                else:
                    return "yea can, wait i create github repo first"
            if any(w in last_user_msg for w in ["eat", "lunch", "dinner", "food"]):
                if self.context_type == "supergroup":
                    return "grabbing quick food first, back in 20"
                elif self.context_type == "group":
                    return "munch or kfc? damn hungry alr sia"
                else:
                    return "can ah, where u wanna eat"

            # Generic questions
            if self.context_type == "supergroup":
                return "can check with team first, should be fine"
            elif self.context_type == "group":
                return "idk sia, maybe ask the others too"
            else:
                return "idk leh, prob search yt or google first"

        # Casual statements
        if any(w in last_user_msg for w in ["tired", "exhausted", "sleep", "dead"]):
            if self.context_type == "supergroup":
                return "get some rest first, can resume tmr"
            elif self.context_type == "group":
                return "same bro gg, brain completely fried sia"
            else:
                return "yea sleep early ah, see u tmr"

        if any(w in last_user_msg for w in ["game", "gacha", "star rail", "genshin"]):
            if self.context_type == "group":
                return "we shld roll on same day sia, good luck"
            else:
                return "noice, hope u get early pity"

        # Generic default acknowledgments
        if self.context_type == "supergroup":
            return random.choice([
                "can ah, noted",
                "tested it alr, seems working on my end",
                "2 week sprint, we should be fine",
                "can proceed first",
            ])
        elif self.context_type == "group":
            return random.choice([
                "Noice",
                "yea honestly gg sia",
                "can can, let me know later",
                "wait ah, let me finish this first",
                "lmao true",
            ])
        else:  # personal_chat
            return random.choice([
                "yea can",
                "can submit now ah, everything looks good",
                "wait otw now, reaching soon",
                "the thing is idk how strict they mark",
                "yea prob, see how it goes",
            ])


class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI-compatible chat completion APIs (OpenAI, DeepSeek, Ollama, vLLM).
    Uses standard library urllib.request for zero external dependencies.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"

    def generate_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload_messages = [{"role": "system", "content": system_prompt}] + messages

        body = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": 0.7,
            "max_tokens": 120,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {e.code}: {error_body}") from e
        except Exception as e:
            raise RuntimeError(f"Network error calling OpenAI endpoint: {e}") from e


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude messages API using urllib.request."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022"

    def generate_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> str:
        url = "https://api.anthropic.com/v1/messages"
        body = {
            "model": self.model,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": 120,
            "temperature": 0.7,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API error {e.code}: {error_body}") from e


class HuggingFaceClient(BaseLLMClient):
    """Client for local open-source weights using transformers pipeline or PEFT LoRA adapter."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ):
        base_model = model_name or "Qwen/Qwen2.5-7B-Instruct"
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
            logger.info("Initializing HuggingFace pipeline on %s (dtype=%s)...", device, dtype)

            if adapter_path:
                logger.info("Loading base model: %s with LoRA adapter: %s", base_model, adapter_path)
                tokenizer_src = adapter_path if Path(adapter_path).is_dir() else base_model
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                )
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
                model = model.merge_and_unload()
                self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=base_model,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
            logger.info("Successfully loaded local HuggingFace model.")
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model '{base_model}' (adapter: {adapter_path}): {e}") from e

    def generate_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> str:
        full_conversation = [{"role": "system", "content": system_prompt}] + messages
        output = self.pipeline(
            full_conversation,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
        )
        # Extract assistant turn
        generated_msg = output[0]["generated_text"][-1]
        return generated_msg["content"].strip()


def create_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    adapter: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    context_type: str = "personal_chat",
) -> BaseLLMClient:
    """Factory to initialize the appropriate LLM client with automatic fallback."""
    provider_clean = (provider or "").strip().lower()

    if adapter or provider_clean in ["hf", "huggingface", "llama", "qwen", "lora"]:
        logger.info("Initializing HuggingFaceClient...")
        return HuggingFaceClient(model_name=model, adapter_path=adapter)

    if provider_clean in ["openai", "chatgpt"] or (not provider_clean and os.environ.get("OPENAI_API_KEY")):
        logger.info("Initializing OpenAIClient...")
        return OpenAIClient(api_key=api_key, base_url=base_url, model=model)

    if provider_clean in ["anthropic", "claude"] or (not provider_clean and os.environ.get("ANTHROPIC_API_KEY")):
        logger.info("Initializing AnthropicClient...")
        return AnthropicClient(api_key=api_key, model=model)

    logger.info("No external LLM key provided. Defaulting to high-fidelity SimulatedClient.")
    return SimulatedClient(context_type=context_type)


# =============================================================================
# Conversational Sandbox Loop
# =============================================================================

class PersonalityChatApp:
    """
    Main conversational sandbox orchestrating the 3-layer architecture:
    1. Multi-Context Profiler (extracts linguistic signatures & Big-5)
    2. Context-Adaptive Prompt Synthesizer (creates dynamic instructions)
    3. Inference Engine with Sliding Window & Refusal Deflection Layer
    """

    CONTEXT_MAP = {
        "1": "personal_chat",
        "2": "group",
        "3": "supergroup",
    }

    CONTEXT_LABELS = {
        "personal_chat": "Personal Chat (DM)",
        "group": "Group Chat",
        "supergroup": "Supergroup Chat",
    }

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        context_type: str = "personal_chat",
        window_size: int = 5,
    ):
        self.profiler = MultiContextProfiler()
        self.synthesizer = SystemPromptSynthesizer(self.profiler)
        self.context_type = context_type
        self.window_size = window_size
        # Each conversational exchange has 2 messages (user + assistant); window_size turns = window_size * 2 messages
        self.history: Deque[Dict[str, str]] = deque(maxlen=self.window_size * 2)

        # Preload context profile and prompt
        self.active_profile: ContextProfile = self.profiler.profile_context(self.context_type)
        self.active_system_prompt: str = self.synthesizer.build_system_prompt(
            profile=self.active_profile,
            context_type=self.context_type,
        )

        self.llm_client = llm_client or create_llm_client(context_type=self.context_type)
        if isinstance(self.llm_client, SimulatedClient):
            self.llm_client.update_context(self.context_type)

    def set_context(self, context_key: str) -> None:
        """Switches active social context dynamically."""
        canonical = self.synthesizer.normalize_context_name(context_key)
        self.context_type = canonical
        self.active_profile = self.profiler.profile_context(canonical)
        self.active_system_prompt = self.synthesizer.build_system_prompt(
            profile=self.active_profile,
            context_type=canonical,
        )
        if isinstance(self.llm_client, SimulatedClient):
            self.llm_client.update_context(canonical)
        label = self.CONTEXT_LABELS.get(canonical, canonical)
        print(f"\n[OK] Switched context to: {label}")

    def respond(self, user_input: str) -> Tuple[str, bool]:
        """
        Processes user message through the Refusal Deflection Layer and LLM.
        Returns: (response_text, was_deflected_boolean)
        """
        clean_input = user_input.strip()
        if not clean_input:
            return "", False

        # 1. Refusal Deflection Layer Check
        if RefusalDeflectionLayer.check_query(clean_input):
            deflection = RefusalDeflectionLayer.get_deflection(self.context_type)
            # Record in sliding window
            self.history.append({"role": "user", "content": clean_input})
            self.history.append({"role": "assistant", "content": deflection})
            return deflection, True

        # 2. Add user message to history
        self.history.append({"role": "user", "content": clean_input})

        # 3. LLM Generation
        try:
            reply = self.llm_client.generate_response(
                system_prompt=self.active_system_prompt,
                messages=list(self.history),
            )
            # Clean reply if it includes prefix like "Dingxuan:"
            if reply.lower().startswith("dingxuan:"):
                reply = reply[len("dingxuan:"):].strip()
        except Exception as err:
            logger.warning("LLM client encountered error: %s. Using in-character fallback.", err)
            reply = "wait ah, wifi lag abit"

        # 4. Add assistant response to sliding window
        self.history.append({"role": "assistant", "content": reply})
        return reply, False

    def print_stats(self) -> None:
        """Displays quantitative metrics and Big-5 scores for current context."""
        m = self.active_profile.metrics
        b = self.active_profile.big_five
        print("\n" + "=" * 60)
        print(f" PROFILE STATS: {self.active_profile.display_name}")
        print("=" * 60)
        print(f"  Turns Analyzed : {m.total_analyzed_turns} (Fallback={self.active_profile.is_fallback})")
        print(f"  Avg Length     : {m.avg_words_per_turn} words / {m.avg_chars_per_turn} chars")
        print(f"  Lowercase Rate : {m.all_lower_ratio * 100:.1f}%")
        print(f"  Punctuation    : !={m.exclamation_ratio*100:.1f}%, ?={m.question_ratio*100:.1f}%, ...={m.ellipsis_ratio*100:.1f}%")
        print(f"  Top Fillers    : {[f[0] for f in m.top_fillers[:6]]}")
        print(f"  Big-5 Profile  : Openness={b.openness:.2f}, Conscientiousness={b.conscientiousness:.2f}, "
              f"Extraversion={b.extraversion:.2f}, Agreeableness={b.agreeableness:.2f}, Neuroticism={b.neuroticism:.2f}")
        print("=" * 60 + "\n")

    def run_interactive_loop(self) -> None:
        """Starts the interactive terminal chat loop."""
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")

        print("=" * 70)
        print("  AI PERSONA - CONTEXT-ADAPTIVE CHAT APPLICATION")
        print("=" * 70)
        print("Select Chat Context for AI:")
        print("  [1] Personal Chat (DM)")
        print("  [2] Group Chat")
        print("  [3] Supergroup Chat")

        choice = input("\nSelect [1-3] (default 1): ").strip()
        context_key = self.CONTEXT_MAP.get(choice, "personal_chat")
        self.set_context(context_key)

        print("\nCommands available:")
        print("  /context [1|2|3] - Switch chat context")
        print("  /stats           - View linguistic & Big-5 personality metrics")
        print("  /prompt          - View synthesized system prompt")
        print("  /clear           - Reset conversation history")
        print("  /exit or /quit   - Exit the chat")
        print("-" * 70)
        print(f"Chatting with AI ({self.active_profile.display_name}). Type your message below.\n")

        while True:
            try:
                user_msg = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting chat. Bye!")
                break

            if not user_msg:
                continue

            # Command handling
            if user_msg.startswith("/"):
                cmd_parts = user_msg[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd in ["exit", "quit", "q"]:
                    print("Exiting chat. Bye!")
                    break
                elif cmd == "context":
                    if not arg:
                        print("Usage: /context [1|2|3] or [dm|group|supergroup]")
                    else:
                        try:
                            self.set_context(arg)
                        except ValueError as err:
                            print(f"[Error]: {err}")
                    continue
                elif cmd == "stats":
                    self.print_stats()
                    continue
                elif cmd == "prompt":
                    print("\n--- ACTIVE SYSTEM PROMPT ---\n")
                    print(self.active_system_prompt)
                    print("\n--- END SYSTEM PROMPT ---\n")
                    continue
                elif cmd == "clear":
                    self.history.clear()
                    print("[OK] Conversation history cleared.")
                    continue
                elif cmd in ["help", "h"]:
                    print("Available commands: /context [1-3], /stats, /prompt, /clear, /exit")
                    continue
                else:
                    print(f"Unknown command: /{cmd}. Type /help for options.")
                    continue

            # Generate reply
            reply, was_deflected = self.respond(user_msg)
            deflect_tag = " [Deflection]" if was_deflected else ""
            print(f"AI {deflect_tag}: {reply}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dingxuan Context-Adaptive Personality Chatbot")
    parser.add_argument(
        "--context",
        "-c",
        default="personal_chat",
        choices=["personal_chat", "group", "supergroup", "1", "2", "3", "dm"],
        help="Initial context setting",
    )
    parser.add_argument(
        "--provider",
        "-p",
        default=None,
        choices=["simulated", "mock", "openai", "anthropic", "hf"],
        help="LLM provider (default: auto-detect from env or fallback to simulated)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Specific model name (e.g. gpt-4o-mini, claude-3-5-sonnet-20241022, meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--adapter",
        "-a",
        default=None,
        help="Path to trained LoRA adapter checkpoint directory (e.g. checkpoints/dingxuan_lora/adapter)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for selected provider",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for OpenAI-compatible endpoint",
    )
    args = parser.parse_args()

    # Normalize context
    ctx = SystemPromptSynthesizer.CONTEXT_ALIASES.get(args.context.lower(), "personal_chat")

    llm_client = create_llm_client(
        provider=args.provider,
        model=args.model,
        adapter=args.adapter,
        api_key=args.api_key,
        base_url=args.base_url,
        context_type=ctx,
    )

    app = PersonalityChatApp(
        llm_client=llm_client,
        context_type=ctx,
        window_size=5,
    )
    app.run_interactive_loop()


if __name__ == "__main__":
    main()
