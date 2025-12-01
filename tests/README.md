# Keyframe Query Compiler - Test Suite

This directory contains comprehensive tests for the keyframe query compiler system.

## Test Organization

### 1. Unit Tests (`test_udf_scoring.py`)
Tests individual UDF (User-Defined Function) implementations to ensure they return correct fractional scores based on the percentage of frames satisfying predicates.

**Key tests:**
- `test_velocity_above_fractional_score`: Verifies velocity predicates return correct fractions
- `test_velocity_above_all_satisfy`: Tests edge case where all frames match
- `test_velocity_above_none_satisfy`: Tests edge case where no frames match
- `test_dist_within_two_obj_fractional`: Tests pairwise distance predicates
- `test_heading_diff_agent_to_agent_fractional`: Tests heading difference calculations
- `test_car_turning_fractional`: Tests rotational velocity predicates
- `test_is_approaching_fractional`: Tests relative motion predicates

### 2. Integration Tests (`test_predicate_evaluation.py`)
Tests the predicate expression evaluation system, focusing on how AND/OR/NOT operations combine fractional scores.

**Key tests:**
- `test_single_atom_evaluation`: Basic atomic predicate evaluation
- `test_and_combines_with_minimum`: Verifies AND uses min() semantics
- `test_or_combines_with_maximum`: Verifies OR uses max() semantics
- `test_not_complements_score`: Verifies NOT returns (1.0 - score)
- `test_nested_expressions`: Tests complex nested boolean expressions
- `test_pairwise_predicates_with_and`: Tests combining multiple object predicates

### 3. End-to-End Tests (`test_end_to_end.py`)
Tests the complete query compilation and execution pipeline from QuerySpec to results.

**Key tests:**
- `test_simple_query_returns_fractional_scores`: Verifies full pipeline preserves fractional scores
- `test_query_with_and_preserves_granular_scores`: Tests multi-predicate queries
- `test_always_constraint_with_fractional_scores`: Tests temporal constraints
- `test_score_details_included`: Verifies result metadata is populated
- `test_score_matches_expected_fraction`: Validates scoring accuracy

## Running Tests

### Quick Start
```bash
# Run all tests
./run_tests.sh all

# Run specific test suites
./run_tests.sh unit
./run_tests.sh integration
./run_tests.sh e2e
```

### Using pytest directly
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_udf_scoring.py -v

# Run a specific test function
pytest tests/test_udf_scoring.py::TestUDFScoring::test_velocity_above_fractional_score -v

# Run tests matching a pattern
pytest tests/ -k "fractional" -v
```

## The Scoring Bug and Fix

### Problem
The original implementation used Python's `all()` and `any()` for AND/OR operations, which treat scores as boolean (truthy/falsy) values. This caused all scores to be either 0.0 or 1.0, losing the fractional information about *how well* a predicate was satisfied.

**Before (buggy):**
```python
# AND operation
return all(evaluate(arg) for arg in args)  # Returns True/False

# OR operation  
return any(evaluate(arg) for arg in args)  # Returns True/False
```

### Solution
Changed the evaluation to preserve fractional scores using min/max semantics:

**After (fixed):**
```python
# AND operation (all conditions must be satisfied)
scores = [evaluate(arg) for arg in args]
return min(scores)  # Returns float in [0.0, 1.0]

# OR operation (at least one condition should be satisfied)
scores = [evaluate(arg) for arg in args]
return max(scores)  # Returns float in [0.0, 1.0]

# NOT operation
score = evaluate(arg)
return 1.0 - score  # Complement the score
```

### Scoring Semantics

**Atomic Predicates:**
- Return the fraction of frames in the window that satisfy the condition
- Example: If 7 out of 10 frames have velocity > 3.0, score = 0.7

**AND (Conjunction):**
- Returns minimum score across all sub-predicates
- Intuition: A conjunction is only as strong as its weakest condition
- Example: AND(score=0.7, score=0.5) = 0.5

**OR (Disjunction):**
- Returns maximum score across all sub-predicates
- Intuition: A disjunction is as strong as its strongest condition
- Example: OR(score=0.7, score=0.5) = 0.7

**NOT (Negation):**
- Returns 1.0 - score
- Intuition: Complete complement of the satisfaction level
- Example: NOT(score=0.7) = 0.3

## Test Requirements

The tests require:
- pytest
- numpy
- pandas
- loguru

Install with:
```bash
pip install pytest numpy pandas loguru
```

## Continuous Testing

Consider integrating these tests into your CI/CD pipeline to catch scoring regressions early.

## Extending Tests

When adding new UDFs or modifying the compiler:

1. **Add unit tests** in `test_udf_scoring.py` for new UDFs
2. **Add integration tests** in `test_predicate_evaluation.py` for new expression types
3. **Add e2e tests** in `test_end_to_end.py` for new query patterns

## Debugging Failed Tests

If tests fail:

1. Run with verbose output: `pytest tests/test_file.py -v -s`
2. Run a single test: `pytest tests/test_file.py::TestClass::test_method -v`
3. Use print statements or pdb for debugging
4. Check that sample data matches expectations
5. Verify UDF implementations return scores in [0.0, 1.0]

