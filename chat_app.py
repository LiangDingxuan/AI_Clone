"""
AI Clone - Interactive Inference Chat Application Entrypoint Wrapper.
Delegates to llm/chat_app.py.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.chat_app import (
    BaseLLMClient,
    AnthropicClient,
    HuggingFaceClient,
    OpenAIClient,
    PersonalityChatApp,
    RefusalDeflectionLayer,
    SimulatedClient,
    create_llm_client,
    main,
)

if __name__ == "__main__":
    main()
