import pickle
import argparse
import pandas as pd
from loguru import logger

from NL.registry import UDFRegistry
from NL.compiler import QueryCompiler


logger.add("runs.log", rotation="1 week")

# Example usage
def example_usage(spec_path, data_path, coverage: float | None = None):
    # Sample data
    df = pd.read_csv(data_path)
    
    # UDF registry
    registry = UDFRegistry(df)

    # Load a sample spec
    spec = pickle.load(open(spec_path, "rb"))
    print("spec: ", spec)

    print("[INFO] QueryCompiler initialized successfully with two-stage search implementation")
    print(f"Available UDFs: {list(registry.get_all_udfs().keys())}")
    
    compiler = QueryCompiler(registry, df, logger, coverage=coverage)
    # Execute query with two-stage search
    results = compiler.execute_query(spec)
    print(results)
    

if __name__ == "__main__":
    # example_usage()
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--coverage", type=float, default=1.0, help="Fraction of frames to scan (0-1]")
    args = parser.parse_args()

    logger.info(f"Running example usage with spec: {args.spec} and data: {args.data}, coverage={args.coverage}")
    example_usage(args.spec, args.data, coverage=args.coverage)