import json
import numpy as np
from pathlib import Path
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple, Optional

class Reranker:
    """Learn keyframe importance weights via active learning."""
    
    def __init__(self, all_results: List[Tuple]):
        self.act_learner = None
        # Get keyframes
        self.keyframe_names = sorted(all_results[0][1]['keyframe_scores'].keys())
        
        # Get predicate names for each keyframe
        self.predicate_names = {}
        for kf_name in self.keyframe_names:
            self.predicate_names[kf_name] = sorted(
                all_results[0][1]['predicate_scores'][kf_name].keys()
            )
        
        # Initialize learner
        self._initialize_learner(all_results)
    
    def _get_features(self, result: Dict, feedback: Optional[Dict] = None) -> np.ndarray:
        """
        Feature vector format:
        X = [
            KF1_score,
            KF1:P1,
            KF1:P2,
            ...,
            KF2_score,
            ...,
        ]
        
        Args:
            result: Result dictionary
            feedback: Optional feedback to apply before feature extraction
        """
        # TODO: Apply detailed abjustment based on feedback
        adjusted_result = result
        
        scores = []
        for kf_name in self.keyframe_names:
            # KF score 
            scores.append(adjusted_result['keyframe_scores'][kf_name])
            # Predicate scores
            pred_scores = adjusted_result['predicate_scores'][kf_name]
            for pred in self.predicate_names[kf_name]:
                scores.append(pred_scores[pred])

        return np.array(scores)
    
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
            X_init.append(self._get_features(result))
            # Bottom k
            result = all_results[-(i+1)][1]
            y_init.append(0)
            X_init.append(self._get_features(result))

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

    def _load_feedback_batch(self, labels_file: str = "ui/label_data/labels.jsonl") -> List[Dict]:
        """Load all feedback from JSONL file."""
        feedback_list = []
        labels_path = Path(labels_file)
        
        if not labels_path.exists():
            print(f"⚠️ No feedback file found at {labels_file}")
            return feedback_list
        
        with open(labels_path, 'r') as f:
            for line in f:
                if line.strip():
                    feedback_list.append(json.loads(line))
        
        print(f"✓ Loaded {len(feedback_list)} feedback entries from {labels_file}")
        return feedback_list
    
    def label_and_learn(self, all_results: List[Tuple],
                       labels_file: str = "ui/label_data/labels.jsonl") -> None:
        """
        Read feedback from JSONL and update the active learner.
        
        Args:
            all_results: List of (dataset_idx, result_dict) tuples
            labels_file: Path to feedback JSONL file
        """
        # Load all feedback
        feedback_batch = self._load_feedback_batch(labels_file)

        if not feedback_batch:
            print("⚠️ No feedback to learn from")
            return
        
        # Prepare training data
        X_train = []
        y_train = []

        for feedback_entry in feedback_batch:
            result_idx = feedback_entry['result_idx']
            result = all_results[result_idx][1]
            overall_rating = feedback_entry['overall_rating']
            
            # Prepare feedback dict for adjustments
            feedback = {
                'keyframe_feedback': feedback_entry.get('keyframe_feedback', {}),
                'predicate_feedback': feedback_entry.get('predicate_feedback', {})
            }
            
            # Extract features with feedback adjustments
            features = self._get_features(result, feedback)
            
            X_train.append(features)
            y_train.append(overall_rating)
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Binarize labels for classification (treat 0.5 as positive for now)
        y_binary = (y_train >= 0.5).astype(int)
        
        # Update the learner
        print(f"\n{'='*70}")
        print("LEARNING FROM FEEDBACK")
        print(f"{'='*70}")
        print(f"Training examples: {len(X_train)}")
        print(f"  Positive: {np.sum(y_binary)}")
        print(f"  Negative: {len(y_binary) - np.sum(y_binary)}")
    
        # Teach the active learner
        self.act_learner.teach(X_train, y_binary)
        
        # Display feature importances
        if hasattr(self.act_learner.estimator, 'feature_importances_'):
            importances = self.act_learner.estimator.feature_importances_
            
            print("\nTop 3 Most Important Features:")
            feature_names = self._get_feature_names()
            importance_pairs = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for feat_name, importance in importance_pairs:
                bar = "█" * int(importance * 50)
                print(f"  {feat_name:30s}: {importance:.4f} {bar}")
        
        print(f"\n✓ Learning complete!")
    
    def _get_feature_names(self) -> List[str]:
        """Get human-readable feature names."""
        names = []
        for kf_name in self.keyframe_names:
            names.append(f"{kf_name}_score")
            for pred_name in self.predicate_names[kf_name]:
                names.append(f"{kf_name}:{pred_name}")
        return names

    def rerank_results(self, all_results: List[Tuple]) -> List[Tuple]:
        """
        Rerank results using the learned model.
        
        Args:
            all_results: List of (dataset_idx, result_dict) tuples
            
        Returns:
            Reranked list of results
        """
        # Extract features for ALL results (no feedback adjustments during inference)
        X = np.array([
            self._get_features(result[1])
            for result in all_results
        ])
        
        # Predict probabilities
        final_probs = self.act_learner.predict_proba(X)[:, 1]
        
        # Sort by probability (descending)
        ranked_indices = np.argsort(final_probs)[::-1]
        ranked_results = [all_results[i] for i in ranked_indices]
        
        return ranked_results
