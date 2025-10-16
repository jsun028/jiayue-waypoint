import pickle
import numpy as np
import pandas as pd
import argparse
from registry import UDFRegistry
from compiler import QueryCompiler
from specs import print_spec_details


# Example usage
def example_usage():
    # Sample data
    df = pd.read_csv('dataset/scene_scene-0225.csv')
    # metadata path
    metadata_path = "metadata/scene_scene-0225_stats.json"
    
    # UDF registry
    registry = UDFRegistry(df)

    # Load a sample spec
    spec = pickle.load(open("spec_modified.pkl", "rb"))
    print_spec_details(spec)

    
    print("[INFO] QueryCompiler initialized successfully with two-stage search implementation")
    print(f"Available UDFs: {list(registry.get_all_udfs().keys())}")
    
    compiler = QueryCompiler(registry, df, metadata_path)    
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Execute query with optional sampling')
    parser.add_argument('--estimation_mode', action='store_true', default=False, help='Enable estimation mode')
    args = parser.parse_args()

    # Set parameters based on arguments
    estimation_mode = args.estimation_mode

    # Print execution mode
    if estimation_mode:
        print(f"[INFO] Executing query with estimation mode...")
    else:
        print(f"[INFO] Executing full search...")

    # Execute query with parameters
    results = compiler.execute_query(spec, estimation_mode=estimation_mode)

    
    print(f"[INFO] Results: {results}")
    

if __name__ == "__main__":
    example_usage()
