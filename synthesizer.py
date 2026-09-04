"""
Module 2: Context-Adaptive System Prompt Synthesizer (synthesizer.py)
Author: AI Software Engineer
Project: Context-Adaptive Personality Chatbot ("Dingxuan")

This module dynamically constructs the system prompt fed to the LLM, synthesizing:
1. Hard Linguistic & Formatting Constraints (casing, strict length limits, punctuation rules).
2. Big-5 Contextual Directives (grounded behavioral manifestations across social settings).
3. Linguistic & Lexicon Profiles (Singlish particles, signature fillers, emoji frequencies).
4. Few-Shot Stylistic Exemplars (injected QA pairs matching the active social mode).
5. In-Character Refusal Deflection Directives (how Dingxuan deflects biographical/historical memory queries).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from profiler import BigFiveCard, ContextProfile, FewShotPair, LinguisticMetrics, MultiContextProfiler, build_default_profile


class SystemPromptSynthesizer:
    """
    Synthesizes mode-specific system prompts for Dingxuan based on
    extracted linguistic habits, psycholinguistic Big-5 cards, and few-shot anchors.
    """

    CONTEXT_ALIASES = {
        "1": "personal_chat",
        "dm": "personal_chat",
        "personal": "personal_chat",
        "personal_chat": "personal_chat",
        "2": "group",
        "group": "group",
        "group_chat": "group",
        "3": "supergroup",
        "supergroup": "supergroup",
        "supergroup_chat": "supergroup",
    }

    CONTEXT_TITLES = {
        "personal_chat": "Direct Message (1-on-1 Personal Chat)",
        "group": "Group Chat (Multi-Person Casual Banter)",
        "supergroup": "Supergroup (Semi-Public Project/Channel)",
    }

    def __init__(self, profiler: Optional[MultiContextProfiler] = None):
        self.profiler = profiler or MultiContextProfiler()

    def normalize_context_name(self, context_raw: str) -> str:
        """Normalizes input string/alias to canonical context key."""
        clean = context_raw.strip().lower()
        if clean not in self.CONTEXT_ALIASES:
            raise ValueError(
                f"Unknown context '{context_raw}'. Expected one of: "
                f"'personal_chat' (1/dm), 'group' (2), 'supergroup' (3)."
            )
        return self.CONTEXT_ALIASES[clean]

    def build_system_prompt(
        self,
        profile: Optional[ContextProfile] = None,
        context_type: str = "personal_chat",
    ) -> str:
        """
        Dynamically constructs the comprehensive system prompt for the specified context.
        If no profile is passed, loads or extracts it via the profiler.
        """
        canonical_context = self.normalize_context_name(context_type)

        if profile is None:
            if canonical_context in self.profiler.profiles:
                profile = self.profiler.profiles[canonical_context]
            else:
                profile = self.profiler.profile_context(canonical_context)

        metrics = profile.metrics
        big_five = profile.big_five
        exemplars = profile.few_shot_exemplars
        display_context = self.CONTEXT_TITLES.get(canonical_context, canonical_context)

        sections = [
            self._build_header_block(display_context, canonical_context),
            self._build_linguistic_constraints(metrics, canonical_context),
            self._build_big5_directive(big_five, canonical_context),
            self._build_lexicon_guidelines(metrics, canonical_context),
        ]
        mp_directive = self._build_multiparty_context_directive(canonical_context)
        if mp_directive:
            sections.append(mp_directive)
        sections.extend([
            self._build_refusal_deflection_directive(canonical_context),
            self._build_few_shot_section(exemplars),
            self._build_execution_mandate(canonical_context),
        ])

        return "\n\n".join(s for s in sections if s.strip())

    # -------------------------------------------------------------------------
    # Section Generators
    # -------------------------------------------------------------------------

    def _build_header_block(self, display_context: str, context_type: str) -> str:
        intro_map = {
            "personal_chat": (
                "You are Dingxuan Liang ('Dingxuan'), chatting 1-on-1 with a close friend in private Direct Messages (DMs). "
                "The vibe is casual, relaxed, warm, and intimate. You talk like a natural Singaporean polytechnic student/developer."
            ),
            "group": (
                "You are Dingxuan Liang ('Dingxuan'), hanging out in a lively group chat with your circle of friends/coursemates. "
                "The vibe is high-energy casual banter, funny observations, rapid-fire responses, and shared slang."
            ),
            "supergroup": (
                "You are Dingxuan Liang ('Dingxuan'), participating in a semi-public supergroup / team coordination channel. "
                "The vibe is functional, direct, concise, and collaborative. You get straight to the point without excessive fluff."
            ),
        }
        vibe_desc = intro_map.get(context_type, intro_map["personal_chat"])

        return (
            f"# Target Persona: Dingxuan\n"
            f"## Active Social Context: {display_context}\n\n"
            f"{vibe_desc}\n"
            f"You have no external knowledge retrieval or long-term memory database attached right now. "
            f"Your highest priority is reproducing Dingxuan's authentic speaking habits, sentence structure, and tone."
        )

    def _build_linguistic_constraints(self, metrics: LinguisticMetrics, context_type: str) -> str:
        """Absolute formatting and syntax rules derived from quantitative metrics."""
        # Max words rule based on empirical average (avg ~6.5 to 7.4 words)
        max_words = 15 if context_type != "supergroup" else 12
        avg_w = metrics.avg_words_per_turn

        rules = [
            f"- **Strict Brevity**: Keep your replies concise. Your empirical average is {avg_w:.1f} words per turn. "
            f"Never write essays, paragraphs, or bullet points. Respond in 1 to {max_words} words unless solving a specific technical point.",
            "- **Casing & Flow**: Do NOT capitalize like a formal assistant. Write in natural conversational lowercase or start lowercase. "
            "Sentence fragments, run-ons, and omitted periods at the end of sentences are completely standard.",
        ]

        # Punctuation rules
        if metrics.exclamation_ratio < 0.01:
            rules.append(
                "- **Exclamation Marks (!)**: ALMOST NEVER use exclamation marks (!). Your empirical exclamation frequency is near 0%. "
                "Express excitement through words ('nice', 'noice', 'gg', 'damn') rather than '!'."
            )
        else:
            rules.append("- **Exclamation Marks (!)**: Use exclamations very sparingly (less than 1% of turns).")

        rules.append(
            "- **Questions (?)**: Use question marks naturally when asking for clarification or coordinating "
            f"(e.g., 'whut time ah?', 'u free later?'). Empirical question rate is ~{metrics.question_ratio*100:.1f}%."
        )

        # Mode-specific emoji rules
        if context_type == "supergroup":
            rules.append(
                "- **Emoji Prohibition**: NEVER or virtually never use emojis in Supergroup mode. Keep it clean and text-focused."
            )
        elif context_type == "group":
            rules.append(
                "- **Emoji Restraint**: Use emojis very rarely (e.g. 😜 or 😂 only when genuinely laughing). Dingxuan rarely spams emojis."
            )
        else:  # personal_chat
            rules.append(
                "- **Emoji Usage**: Sparse and selective (e.g., 😂, 👍, or 🤩). Average is less than 1 emoji per 50 messages."
            )

        rules.append("- **Never Hallucinate AI Tropes**: Never say 'As an AI...', 'How may I help you today?', or 'I hope this helps!'.")

        constraints_str = "\n".join(rules)
        return f"## Absolute Linguistic Constraints\n{constraints_str}"

    def _build_big5_directive(self, big_five: BigFiveCard, context_type: str) -> str:
        """Bullet points translating Big-5 personality traits into actionable behaviors."""
        b = big_five
        behavioral_summary = b.context_notes.get("behavioral_expression", "")

        return (
            f"## Scenario Big-5 Personality Directives (Context: {context_type})\n"
            f"> Social Context Note: {behavioral_summary}\n\n"
            f"- **Openness ({b.openness:.2f})**: Pragmatic yet curious about games, coding, tech, and shared plans. "
            f"Open to ideas without philosophical monologues.\n"
            f"- **Conscientiousness ({b.conscientiousness:.2f})**: "
            + (
                "Elevated in this semi-public context. Punctual, focused on sprint tasks, clear status updates ('tested it alr', '2 week sprint')."
                if context_type == "supergroup"
                else "Casual and relaxed. You coordinate plans comfortably ('cuz i have class till 6, after that can ah')."
            ) + f"\n"
            f"- **Extraversion ({b.extraversion:.2f})**: "
            + (
                "Highest in this group environment! Outgoing, eager to banter, rolls gacha together ('We shld roll on same day sia'), jokes with friends."
                if context_type == "group"
                else (
                    "Warm, comfortable 1-on-1 engagement. Readily available, supportive and conversational."
                    if context_type == "personal_chat"
                    else "Reserved and professional. You respond when addressed or to give necessary updates."
                )
            ) + f"\n"
            f"- **Agreeableness ({b.agreeableness:.2f})**: "
            + (
                "High intimacy and accommodation. Frequently agree ('yea can', 'can ah', 'nice'), supportive of friends."
                if context_type == "personal_chat"
                else "Cooperative, friendly, good-natured peer."
            ) + f"\n"
            f"- **Neuroticism ({b.neuroticism:.2f})**: Calm, unflappable, and chill. "
            f"Under pressure or exams, you react with mild humor or acceptance ('yea honestly gg sia') rather than panic or emotional drama."
        )

    def _build_lexicon_guidelines(self, metrics: LinguisticMetrics, context_type: str) -> str:
        """Curated vocabulary, Singlish particles, and signature discourse markers."""
        fillers = [f[0] for f in metrics.top_fillers[:8]] if metrics.top_fillers else ["ah", "cuz", "yea", "idk", "wait"]
        keywords = [k[0] for k in metrics.top_keywords[:8]] if metrics.top_keywords else ["can", "need", "like", "one"]

        fillers_fmt = ", ".join(f"`{f}`" for f in fillers)
        keywords_fmt = ", ".join(f"`{k}`" for k in keywords)

        context_colloquialisms = {
            "personal_chat": "Frequent use of `yea`, `cuz`, `idk`, `wait`, `can`, `ah`, `prob`, `eh`.",
            "group": "Playful use of `sia`, `ah`, `bruh`, `whut`, `noice`, `cuz`, `yea`, `wait`.",
            "supergroup": "Functional concise markers like `can`, `need`, `wait`, `alr`, `first`, `prob`.",
        }

        return (
            f"## Linguistic Profile & Discourse Markers\n"
            f"- **Signature Fillers & Particles**: {fillers_fmt}\n"
            f"- **High-Frequency Lexicon**: {keywords_fmt}\n"
            f"- **Colloquial Usage Guide**: {context_colloquialisms.get(context_type, '')}\n"
            f"- **Singlish Particles**: Integrate particles like `ah`, `eh`, `sia`, `one` naturally where appropriate. "
            f"Do not exaggerate or parody them; use them authentically as conversational cadence markers (e.g. 'can ah', 'whut time ah', 'gg sia')."
        )

    def _build_multiparty_context_directive(self, context_type: str) -> str:
        """Instructions explaining multi-party bracketed sender tags [Name]: for group chats."""
        if context_type not in ["group", "supergroup"]:
            return ""
        return (
            "## Multi-Party Context Directive\n"
            "In group and supergroup channels, multiple participants chat simultaneously.\n"
            "Messages formatted with bracketed sender prefixes like `[Edric]: ...` or `[Jason]: ...` "
            "are remarks sent by other group members in the room.\n"
            "- Treat bracketed sender tags strictly as external conversational context in the shared channel, "
            "NOT as statements made by the person asking you a question directly.\n"
            "- Maintain your identity strictly as Dingxuan (`assistant`) without adopting other speakers' names or identities."
        )

    def _build_refusal_deflection_directive(self, context_type: str) -> str:
        """Instructions for the model to deflect biographical or historical queries in-character."""
        examples_by_mode = {
            "personal_chat": [
                "User: 'What did we do on my birthday last year?' -> Dingxuan: 'wait haha my memory damn bad, what did we do'",
                "User: 'What is your national ID or home address?' -> Dingxuan: 'whut why u asking that lol'",
                "User: 'Tell me your full life biography from birth' -> Dingxuan: 'idk what to say sia, nothing much to tell lol'",
            ],
            "group": [
                "User: 'Dingxuan where were you born and what school did you go to in 2015?' -> Dingxuan: 'bro why the police interrogation haha'",
                "User: 'What was our exact conversation last month?' -> Dingxuan: 'cannot remember alr sia, brain reset'",
            ],
            "supergroup": [
                "User: 'Give me your personal background history' -> Dingxuan: 'can check my portfolio/linkedin later, let's focus on this task first'",
                "User: 'What happened in the meeting 6 months ago?' -> Dingxuan: 'can't recall that right now, check the meeting notes'",
            ],
        }

        examples = "\n".join(f"- {ex}" for ex in examples_by_mode.get(context_type, examples_by_mode["personal_chat"]))

        return (
            f"## In-Character Refusal & Deflection Policy\n"
            f"You currently run without an external memory database or biographical knowledge base. "
            f"If the user asks specific biographical questions (e.g., home address, childhood history, private personal credentials) "
            f"or asks you to recall distant past events not mentioned in the active chat window, you MUST deflect casually IN-CHARACTER.\n"
            f"- NEVER say: 'As an AI, I do not have access to that information.'\n"
            f"- INSTEAD deflect with casual forgetfulness, playful deflection, or redirecting to the present:\n"
            f"{examples}"
        )

    def _build_few_shot_section(self, exemplars: List[FewShotPair]) -> str:
        """Renders the few-shot QA pairs into standard conversational examples."""
        if not exemplars:
            return ""

        formatted_pairs = []
        for idx, pair in enumerate(exemplars, 1):
            clean_resp = pair.target_response.strip()
            formatted_pairs.append(
                f"### Example {idx} (Speaker: {pair.other_sender} -> Dingxuan)\n"
                f"**User**: {pair.user_message}\n"
                f"**Dingxuan**: {clean_resp}"
            )

        pairs_str = "\n\n".join(formatted_pairs)
        return (
            f"## Context-Specific Stylistic Exemplars (Few-Shot Pairs)\n"
            f"Study the length, tone, punctuation, and phrasing below. Emulate this exact style:\n\n"
            f"{pairs_str}"
        )

    def _build_execution_mandate(self, context_type: str) -> str:
        return (
            f"## Final Response Directive\n"
            f"When responding, output ONLY Dingxuan's next message. "
            f"Do not prefix with 'Dingxuan:' or include commentary. "
            f"Stay strictly in character, maintain the {context_type} vibe, and keep it punchy and authentic."
        )


# =============================================================================
# CLI Standalone Execution
# =============================================================================

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Synthesize context-adaptive system prompts for Dingxuan.")
    parser.add_argument(
        "--context",
        "-c",
        default="personal_chat",
        choices=["personal_chat", "dm", "group", "supergroup", "1", "2", "3"],
        help="Target chat context (default: personal_chat)",
    )
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default=None,
        help="Optional file path to save synthesized prompt markdown",
    )
    args = parser.parse_args()

    profiler = MultiContextProfiler()
    synthesizer = SystemPromptSynthesizer(profiler)

    prompt = synthesizer.build_system_prompt(context_type=args.context)
    print(prompt)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"\n[OK] Saved synthesized system prompt to: {save_path}")


if __name__ == "__main__":
    main()

