import pickle
import numpy as np
import pandas as pd
from registry import UDFRegistry
from compiler import QueryCompiler

# Example usage
def example_usage():
    # Sample data
    df = pd.read_csv('dataset/scene-0357_899_ego.jsonl.csv')
    
    # UDF registry
    registry = UDFRegistry(df)

    # Load a sample spec
    spec = pickle.load(open("spec.pkl", "rb"))
    print("spec: ", spec)

    print("[INFO] QueryCompiler initialized successfully with two-stage search implementation")
    print(f"Available UDFs: {list(registry.get_all_udfs().keys())}")
    
    compiler = QueryCompiler(registry, df)
    # Execute query with two-stage search
    results = compiler.execute_query(spec)
    print(results)
    

if __name__ == "__main__":
    example_usage()
