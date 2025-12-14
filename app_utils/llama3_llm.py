from typing import Any, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import subprocess

class Llama3LLM(LLM):
    model: str = "llama3:latest"
    temperature: float = 0.2
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "llama3_local"

    # -----------------------------------------------------------
    # Ollama local call
    # -----------------------------------------------------------
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        try:
            # Run Ollama locally without --json
            process = subprocess.Popen(
                ["ollama", "run", self.model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            # Send the prompt via stdin
            stdout, stderr = process.communicate(input=prompt, timeout=60)

            if process.returncode != 0:
                return f"Ollama error: {stderr.strip()}"

            # Return the raw text output
            return stdout.strip()

        except subprocess.TimeoutExpired:
            process.kill()
            return "Ollama process timed out."
        except Exception as exc:
            return f"Error calling local Llama 3 through Ollama: {str(exc)}"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
