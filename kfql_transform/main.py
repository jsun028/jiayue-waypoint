import argparse, pathlib, time
from config import LLMConfig, TransformConfig
from pipeline import TransformPipeline

def main(argv=None):
    p = argparse.ArgumentParser(description="Transform JSON annotations → KeyframeQL query via LLM")
    p.add_argument("--input", "-i", type=pathlib.Path, required=True, help="Path to JSON annotations")
    p.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    p.add_argument("--gpt-model", default="gpt-5")
    p.add_argument("--gemini-model", default="gemini-1.5-pro")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-output-tokens", type=int, default=4000)
    p.add_argument("--reasoning-effort", default="high")
    p.add_argument("--verbosity", default="low")

    # BooleanOptionalAction: --examples / --no-examples, by default True
    p.add_argument(
        "--examples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include few-shot examples (default: True). Use --no-examples to disable."
    )
    p.add_argument(
        "--query-dsl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include query DSL (default: True). Use --no-query-dsl to disable."
    )

    p.add_argument("--fast-vel", type=float, default=2.0)

    args = p.parse_args(argv)
    print("args: \n", args)

    annotations = args.input.read_text(encoding="utf-8")
    print("============= annotations: \n", annotations)

    llm_cfg = LLMConfig(
        provider=args.provider,
        openai_model=args.gpt_model,
        gemini_model=args.gemini_model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        openai_reasoning_effort=args.reasoning_effort,
        openai_verbosity=args.verbosity,
    )
    xform_cfg = TransformConfig(
        include_examples=args.examples,       # by default, True
        include_query_dsl=args.query_dsl,   # by default, True
        default_fast_velocity=args.fast_vel,
    )

    # Evaluate runtime
    start_time = time.perf_counter()

    pipeline = TransformPipeline(llm_cfg, xform_cfg)
    res = pipeline.run(annotations)
    end_time = time.perf_counter()
    print(f"Runtime: {end_time - start_time:.2f} seconds")
        
    # print("# ==== RAW MODEL OUTPUT ====\n") # for debugging
    # print(res.raw_text)
    print("\n\n# ==== EXTRACTED PYTHON QUERY ====\n")
    print(res.code)

if __name__ == "__main__":
    main()