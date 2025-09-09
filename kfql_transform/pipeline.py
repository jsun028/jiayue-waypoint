from dataclasses import dataclass
from config import LLMConfig, TransformConfig
from prompts import build_prompt
from providers import OpenAIProvider, GeminiProvider
from validators import ensure_json_str
from extract import extract_python_block

@dataclass
class PipelineResult:
    raw_text: str
    code: str

class TransformPipeline:
    def __init__(self, llm_cfg: LLMConfig, xform_cfg: TransformConfig):
        self.llm_cfg = llm_cfg
        self.xform_cfg = xform_cfg

        if llm_cfg.provider == "openai":
            self.provider = OpenAIProvider(
                model=llm_cfg.openai_model,
                reasoning_effort=llm_cfg.openai_reasoning_effort,
                verbosity=llm_cfg.openai_verbosity,
            )
        elif llm_cfg.provider == "gemini":
            self.provider = GeminiProvider(model=llm_cfg.gemini_model)
        else:
            raise ValueError(f"Unknown provider: {llm_cfg.provider}")

    def run(self, annotations: str | dict) -> PipelineResult:
        json_str = ensure_json_str(annotations)
        prompt = build_prompt(
            user_command=self.xform_cfg.command,
            json_annotations=json_str,
            fast_velocity=self.xform_cfg.default_fast_velocity,
            include_query_dsl=self.xform_cfg.include_query_dsl,
            include_examples=self.xform_cfg.include_examples,
        )

        print("============= prompt: \n", prompt)

        raw = self.provider.generate(
            prompt,
            temperature=self.llm_cfg.temperature,
            max_output_tokens=self.llm_cfg.max_output_tokens,
            timeout_s=self.llm_cfg.timeout_s,
        )
        code = extract_python_block(raw) if self.xform_cfg.require_code_fence else raw.strip()
        return PipelineResult(raw_text=raw, code=code)
