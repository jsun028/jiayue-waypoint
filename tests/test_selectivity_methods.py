#!/usr/bin/env python3
"""
Comprehensive test suite for selectivity estimation methods.

This script tests Method 1 (Correlation-Adjusted) and Method 2 (Log-Linear Blending)
with various test cases to validate correctness and demonstrate usage patterns.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'NL'))

from NL.optimizer.selectivity_integration import correlated_and_multi, blended_and_multi, correlated_and_matrix
import numpy as np


def test_basic_functionality():
    """Test basic functionality with simple cases"""
    print("=" * 80)
    print("TEST 1: Basic Functionality")
    print("=" * 80)
    
    # Test case 1: Simple two predicates
    sels_2 = [0.5, 0.4]
    naive_2 = sels_2[0] * sels_2[1]
    
    print(f"Two predicates: {sels_2}")
    print(f"Naive product: {naive_2:.4f}")
    print(f"Correlated (ρ=0.5): {correlated_and_multi(sels_2, rho=0.5):.4f}")
    print(f"Blended (β=3.0): {blended_and_multi(sels_2, alpha=1.0, beta=3.0):.4f}")
    print()
    
    # Test case 2: Three predicates (from user's example)
    sels_3 = [0.25, 0.43, 0.35]
    naive_3 = sels_3[0] * sels_3[1] * sels_3[2]
    
    print(f"Three predicates: {sels_3}")
    print(f"Naive product: {naive_3:.4f}")
    print(f"Correlated (ρ=0.6): {correlated_and_multi(sels_3, rho=0.6):.4f}")
    print(f"Blended (β=3.0): {blended_and_multi(sels_3, alpha=1.0, beta=3.0):.4f}")
    print()


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("=" * 80)
    print("TEST 2: Edge Cases")
    print("=" * 80)
    
    # Empty list
    print("Empty list:")
    print(f"Correlated: {correlated_and_multi([]):.4f}")
    print(f"Blended: {blended_and_multi([]):.4f}")
    print()
    
    # Single predicate
    print("Single predicate [0.3]:")
    print(f"Correlated: {correlated_and_multi([0.3]):.4f}")
    print(f"Blended: {blended_and_multi([0.3]):.4f}")
    print()
    
    # Very low selectivities
    low_sels = [0.01, 0.02, 0.03]
    print(f"Very low selectivities {low_sels}:")
    print(f"Naive: {np.prod(low_sels):.6f}")
    print(f"Correlated (ρ=0.5): {correlated_and_multi(low_sels, rho=0.5):.6f}")
    print(f"Blended (β=3.0): {blended_and_multi(low_sels, alpha=1.0, beta=3.0):.6f}")
    print()
    
    # High selectivities
    high_sels = [0.8, 0.9, 0.85]
    print(f"High selectivities {high_sels}:")
    print(f"Naive: {np.prod(high_sels):.4f}")
    print(f"Correlated (ρ=0.5): {correlated_and_multi(high_sels, rho=0.5):.4f}")
    print(f"Blended (β=3.0): {blended_and_multi(high_sels, alpha=1.0, beta=3.0):.4f}")
    print()


def test_correlation_parameter_sweep():
    """Test correlation method with different rho values"""
    print("=" * 80)
    print("TEST 3: Correlation Parameter Sweep")
    print("=" * 80)
    
    sels = [0.25, 0.43, 0.35]
    naive = np.prod(sels)
    
    print(f"Selectivities: {sels}")
    print(f"Naive product: {naive:.4f}")
    print()
    
    print("Correlation coefficient (ρ) sweep:")
    for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        result = correlated_and_multi(sels, rho=rho)
        print(f"ρ = {rho:.1f}: {result:.4f}")
    
    print()


def test_blending_parameter_sweep():
    """Test blending method with different alpha and beta values"""
    print("=" * 80)
    print("TEST 4: Blending Parameter Sweep")
    print("=" * 80)
    
    sels = [0.25, 0.43, 0.35]
    naive = np.prod(sels)
    
    print(f"Selectivities: {sels}")
    print(f"Naive product: {naive:.4f}")
    print()
    
    print("Beta (β) parameter sweep (α=1.0):")
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
        result = blended_and_multi(sels, alpha=1.0, beta=beta)
        print(f"β = {beta:4.1f}: {result:.4f}")
    
    print()
    print("Alpha (α) parameter sweep (β=3.0):")
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        result = blended_and_multi(sels, alpha=alpha, beta=3.0)
        print(f"α = {alpha:3.1f}: {result:.4f}")
    
    print()


def test_correlation_matrix():
    """Test correlation matrix functionality"""
    print("=" * 80)
    print("TEST 5: Correlation Matrix")
    print("=" * 80)
    
    sels = [0.25, 0.43, 0.35]
    naive = np.prod(sels)
    
    print(f"Selectivities: {sels}")
    print(f"Naive product: {naive:.4f}")
    print()
    
    # Test case 1: Uniform correlations
    print("Uniform correlation matrix:")
    uniform_matrix = {
        (0, 1): 0.5,
        (0, 2): 0.5,
        (1, 2): 0.5
    }
    result_uniform = correlated_and_matrix(sels, uniform_matrix)
    result_uniform_method = correlated_and_multi(sels, rho=0.5)
    print(f"Matrix result: {result_uniform:.4f}")
    print(f"Uniform ρ=0.5: {result_uniform_method:.4f}")
    print(f"Difference: {abs(result_uniform - result_uniform_method):.6f}")
    print()
    
    # Test case 2: Varied correlations
    print("Varied correlation matrix:")
    varied_matrix = {
        (0, 1): 0.2,  # low correlation
        (0, 2): 0.8,  # high correlation
        (1, 2): 0.4   # moderate correlation
    }
    result_varied = correlated_and_matrix(sels, varied_matrix)
    print(f"Varied matrix: {result_varied:.4f}")
    print(f"Matrix details: {varied_matrix}")
    print()
    
    # Test case 3: Missing correlations (should default to 0)
    print("Partial correlation matrix:")
    partial_matrix = {
        (0, 1): 0.6,  # only correlation between first two predicates
        # (0, 2) and (1, 2) missing - should default to 0
    }
    result_partial = correlated_and_matrix(sels, partial_matrix)
    print(f"Partial matrix: {result_partial:.4f}")
    print(f"Matrix details: {partial_matrix}")
    print()


def test_realistic_scenarios():
    """Test with realistic KeyframeQL scenarios"""
    print("=" * 80)
    print("TEST 6: Realistic KeyframeQL Scenarios")
    print("=" * 80)
    
    # Scenario 1: Car detection with velocity and position constraints
    print("Scenario 1: Car with velocity > 2.0 AND yaw in [-0.5, 0.5]")
    car_scenario = [0.3, 0.4, 0.6]  # velocity, yaw_low, yaw_high
    print(f"Individual selectivities: {car_scenario}")
    print(f"Naive: {np.prod(car_scenario):.4f}")
    print(f"Correlated (ρ=0.4): {correlated_and_multi(car_scenario, rho=0.4):.4f}")
    print(f"Blended (β=2.5): {blended_and_multi(car_scenario, alpha=1.0, beta=2.5):.4f}")
    print()
    
    # Scenario 2: Person detection with multiple constraints
    print("Scenario 2: Person with multiple spatial constraints")
    person_scenario = [0.2, 0.3, 0.25, 0.4]  # class, x_range, y_range, visibility
    print(f"Individual selectivities: {person_scenario}")
    print(f"Naive: {np.prod(person_scenario):.4f}")
    print(f"Correlated (ρ=0.6): {correlated_and_multi(person_scenario, rho=0.6):.4f}")
    print(f"Blended (β=3.0): {blended_and_multi(person_scenario, alpha=1.0, beta=3.0):.4f}")
    print()
    
    # Scenario 3: Multi-object interaction
    print("Scenario 3: Two cars with distance and relative velocity constraints")
    interaction_scenario = [0.3, 0.4, 0.2, 0.5]  # car1_class, car2_class, distance, rel_velocity
    print(f"Individual selectivities: {interaction_scenario}")
    print(f"Naive: {np.prod(interaction_scenario):.4f}")
    print(f"Correlated (ρ=0.7): {correlated_and_multi(interaction_scenario, rho=0.7):.4f}")
    print(f"Blended (β=4.0): {blended_and_multi(interaction_scenario, alpha=1.0, beta=4.0):.4f}")
    print()


def test_performance_comparison():
    """Compare performance characteristics"""
    print("=" * 80)
    print("TEST 7: Performance Characteristics")
    print("=" * 80)
    
    # Test with increasing number of predicates
    base_sels = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("Increasing number of predicates:")
    print("Predicates | Naive    | Correlated | Blended")
    print("-" * 45)
    
    for n in range(1, 6):
        sels = base_sels[:n]
        naive = np.prod(sels)
        correlated = correlated_and_multi(sels, rho=0.5)
        blended = blended_and_multi(sels, alpha=1.0, beta=3.0)
        
        print(f"{n:10d} | {naive:.4f}   | {correlated:.4f}    | {blended:.4f}")
    
    print()


def test_mathematical_properties():
    """Test mathematical properties and invariants"""
    print("=" * 80)
    print("TEST 8: Mathematical Properties")
    print("=" * 80)
    
    sels = [0.25, 0.43, 0.35]
    
    # Property 1: Monotonicity with respect to correlation
    print("Property 1: Monotonicity with correlation")
    print("As ρ increases, correlated result should increase:")
    prev_result = correlated_and_multi(sels, rho=0.0)
    print(f"ρ = 0.0: {prev_result:.4f}")
    
    for rho in [0.2, 0.4, 0.6, 0.8, 1.0]:
        result = correlated_and_multi(sels, rho=rho)
        print(f"ρ = {rho:.1f}: {result:.4f} (Δ = {result - prev_result:+.4f})")
        prev_result = result
    
    print()
    
    # Property 2: Bounds checking
    print("Property 2: Bounds checking")
    extreme_sels = [0.001, 0.999]
    print(f"Extreme selectivities: {extreme_sels}")
    
    correlated_extreme = correlated_and_multi(extreme_sels, rho=0.5)
    blended_extreme = blended_and_multi(extreme_sels, alpha=1.0, beta=3.0)
    
    print(f"Correlated result: {correlated_extreme:.6f} (should be in [0,1])")
    print(f"Blended result: {blended_extreme:.6f} (should be in [0,1])")
    print(f"Both in bounds: {0 <= correlated_extreme <= 1 and 0 <= blended_extreme <= 1}")
    print()


def test_integration_examples():
    """Show integration examples with SelectivityIntegration class"""
    print("=" * 80)
    print("TEST 9: Integration Examples")
    print("=" * 80)
    
    print("Example 1: Default correlated method")
    print("```python")
    print("integration = SelectivityIntegration(")
    print("    metadata_path='path/to/metadata.json',")
    print("    df=your_dataframe,")
    print("    registry=your_registry,")
    print("    selectivity_method='correlated',  # default")
    print("    correlation_rho=0.5")
    print(")")
    print("```")
    print()
    
    print("Example 2: Blended method with custom parameters")
    print("```python")
    print("integration = SelectivityIntegration(")
    print("    metadata_path='path/to/metadata.json',")
    print("    df=your_dataframe,")
    print("    registry=your_registry,")
    print("    selectivity_method='blended',")
    print("    blending_alpha=1.0,")
    print("    blending_beta=3.0")
    print(")")
    print("```")
    print()
    
    print("Example 3: Advanced correlation matrix")
    print("```python")
    print("# Define pairwise correlations between predicate types")
    print("correlation_matrix = {")
    print("    (0, 1): 0.4,  # velocity vs yaw correlation")
    print("    (0, 2): 0.8,  # velocity vs position correlation")
    print("    (1, 2): 0.2   # yaw vs position correlation")
    print("}")
    print("")
    print("integration = SelectivityIntegration(")
    print("    metadata_path='path/to/metadata.json',")
    print("    df=your_dataframe,")
    print("    registry=your_registry,")
    print("    selectivity_method='correlated',")
    print("    correlation_matrix=correlation_matrix")
    print(")")
    print("```")
    print()


def run_all_tests():
    """Run all test cases"""
    print("KEYFRAMEQL SELECTIVITY ESTIMATION METHODS - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    test_basic_functionality()
    test_edge_cases()
    test_correlation_parameter_sweep()
    test_blending_parameter_sweep()
    test_correlation_matrix()
    test_realistic_scenarios()
    test_performance_comparison()
    test_mathematical_properties()
    test_integration_examples()
    
    print("=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
