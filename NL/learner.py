import ast
import os
import numpy as np
import pandas as pd
from pathlib import Path
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple
from NL.utils.viz import _generate_visualizations

class Reranker:
    """Learn keyframe importance weights via active learning."""
    
    def __init__(self, all_results: List[Tuple]):
        self.act_learner = None
        # Get keyframes
        self.keyframe_names = sorted(all_results[0][1]['keyframe_scores'].keys())
        # print(f"[DEBUG] Keyframes: {self.keyframe_names}")
        
        # Initialize learner
        self._initialize_learner(all_results)
    
    def _get_keyframe_scores(self, keyframe_scores: Dict[str, float]) -> np.ndarray:
        """Extract feature vector from keyframe scores."""
        return np.array([
            keyframe_scores.get(kf_name, 0.0) 
            for kf_name in self.keyframe_names
        ])
    
    def _initialize_learner(self, all_results: List[Tuple], k=5):
        # Initialization: Assume top-k results are relevant, 
        # bottom-k results are not
        k = min(k, len(all_results) // 2)
        y_init = []
        X_init = []
        for i in range(k):
            # Top k
            result = all_results[i][1]
            y_init.append(1)
            X_init.append(self._get_keyframe_scores(result['keyframe_scores']))
            # Bottom k
            result = all_results[-(i+1)][1]
            y_init.append(0)
            X_init.append(self._get_keyframe_scores(result['keyframe_scores']))

        self.act_learner = ActiveLearner(
            estimator=RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                class_weight="balanced",
                random_state=42
            ),
            X_training=np.array(X_init),
            y_training=np.array(y_init)
        )

    def label_and_learn(self, all_results: List[Tuple],
                           data_files: List[str],
                           viz_dir: str,
                           max_iterations: int = 5) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Active learning loop to learn keyframe importance.
        
        Args:
            all_results: List of dicts with 'keyframe_scores' 
            data_files: Path to video
            viz_dir: Viz result folder 
            max_iterations: Max labeling rounds
            
        Returns:
            - learned_scores: Relevance score for each result
            - learned_weights: Dict mapping keyframe_name -> importance weight
        """
        # Extract features for ALL results
        X = np.array([
            self._get_keyframe_scores(result[1]['keyframe_scores'])
            for result in all_results
        ])

        # Active learning loop
        seen_ids = set()
        iteration = 0
        user_termination = False

        # Create viz directory
        out_dir = Path(viz_dir) if viz_dir else Path(viz_dir).with_suffix("").with_name(Path(viz_dir).stem + "_viz") if viz_dir else Path("viz_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        while iteration < max_iterations and not user_termination:
            iteration += 1
            
            probs = self.act_learner.predict_proba(X)[:, 1]
            uncertainty = np.abs(probs - 0.5)
            unseen_mask = np.array([i not in seen_ids for i in range(len(all_results))])
            candidate_uncertainties = np.where(unseen_mask, uncertainty, np.inf)
            
            n_query = min(3, np.sum(unseen_mask))
            if n_query == 0:
                print("\n✓ All examples labeled")
                break
            
            query_indices = np.argsort(candidate_uncertainties)[:n_query]
            
            # Display results
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'='*70}\n")
            
            labels = []
            labeled_queries = []
            for i, idx in enumerate(query_indices):
                print(f"\n--- Result {i+1}/{n_query} (ID: {idx}) ---")
                print(f"Model confidence: {probs[idx]:.3f}")
                print("Keyframe scores:")
                for kf_name in self.keyframe_names:
                    score = all_results[idx][1]['keyframe_scores'][kf_name]
                    bar = "█" * int(score * 20)
                    print(f"  {kf_name}: {score:.3f} {bar}")
                
                self._display_result(all_results, data_files, idx, out_dir, f"result{i+1}")
                
            
                # Get labels
                labels_input = input(f"\nProvide binary labels ('0' or '1') or type 'exit': ")
            
                if labels_input.strip().lower() == 'exit':
                    user_termination = True
                    break
                elif labels_input.strip() in ['0', '1']:    
                    labels.append(ast.literal_eval(labels_input))
                    labeled_queries.append(idx)
                else:
                    print(f"⚠️ Invalid format, skipping") 
                    continue
                
            
            # Update model
            if len(labels) > 0:
                self.act_learner.teach(X[labeled_queries], labels)
                print(f"✓ Learned from {len(labels)} examples")
                seen_ids.update(labeled_queries)
        
        print(f"\n{'='*70}")
        print("LEARNING COMPLETE")
        print(f"{'='*70}")
        print(f"Labeled: {len(seen_ids)} examples")
        
    
    def rerank_results(self, all_results: List[Tuple]):
        # Extract features for ALL results
        X = np.array([
            self._get_keyframe_scores(result[1]['keyframe_scores'])
            for result in all_results
        ])
        final_probs = self.act_learner.predict_proba(X)[:, 1]
        ranked_indices = np.argsort(final_probs)[::-1]
        ranked_results = [all_results[i] for i in ranked_indices]
        return ranked_results


    def _display_result(self, all_results: List[Tuple], data_files: List[str], 
                        idx: int, out_dir: Path, fname: str):
        dataset_idx = all_results[idx][0]
        df = pd.read_csv(data_files[dataset_idx])
        print(f"Visualization: {out_dir}/{fname}.gif")
        _generate_visualizations(df, [all_results[idx][1]], out_dir, top_k=1, fname=fname)
