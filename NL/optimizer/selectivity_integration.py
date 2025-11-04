from NL.optimizer.selectivity_estimator import SelectivityEstimator
import numpy as np
from NL.registry import UDFRegistry
from NL.specs import PredicateAtom, PredicateExpr
import inspect
import numpy as np
import pandas as pd

class SelectivityIntegration:
    
    def __init__(
        self, 
        metadata_path: str, 
        df: pd.DataFrame, 
        registry: UDFRegistry, 
        debug: bool = True, # class-level DEBUG flag (set to True to enable debug printing)
        selectivity_method: str = "correlated", 
        correlation_rho: float = 0.5, 
        blending_alpha: float = 1.0, 
        blending_beta: float = 3.0, 
        correlation_matrix: dict = None):
        """Initialize selectivity integration.
        
        Args:
            metadata_path: Path to metadata JSON file for histogram-based estimation
            df: DataFrame with scene data for UDF-based estimation
            registry: UDF registry to look up and call UDF functions
            debug: Enable debug printing (defaults to class DEBUG flag)
            selectivity_method: Method for combining selectivities ("naive", "correlated", "blended")
            correlation_rho: Average correlation coefficient for correlated method (0-1)
            blending_alpha: Baseline term for blended method
            blending_beta: Scaling term for blended method (higher = less aggressive shrink)
            correlation_matrix: Optional pairwise correlation matrix for correlated method
        """
        self.est = SelectivityEstimator(metadata_path)
        self.df = df
        self.registry = registry
        self.debug = debug if debug is not None else self.DEBUG
        
        # Selectivity combination method configuration
        self.selectivity_method = selectivity_method
        self.correlation_rho = correlation_rho
        self.blending_alpha = blending_alpha
        self.blending_beta = blending_beta
        self.correlation_matrix = correlation_matrix

    def estimate_keyframe_selectivity(self, keyframe_spec, query_spec=None):
        """Estimate selectivity for a keyframe using both histogram and UDF-based methods.
        
        Args:
            keyframe_spec: Keyframe specification containing a PredicateExpr
            query_spec: Full query specification containing object aliases and classes
            
        Returns:
            Dictionary with selectivity estimate
        """
        preds = keyframe_spec.where
        est_result = {}
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"Estimating selectivity for keyframe: {keyframe_spec.name}")
            print(f"{'='*60}")

        def handle_atom(atom: PredicateAtom):
            """Handle a single predicate atom.
            
            Tries histogram-based estimation first, falls back to UDF-based estimation.
            """
            t = atom.type
            
            # Try histogram-based estimation for known single-object predicates
            if t == "velocity_above":
                sel = self.est._hist_fraction(
                    self.est.stats["attribute_histograms"]["velocity_mag"],
                    threshold=atom.value,
                    op=">",
                )
                return sel
            elif t == "velocity_below":
                sel = self.est._hist_fraction(
                    self.est.stats["attribute_histograms"]["velocity_mag"],
                    threshold=atom.value,
                    op="<",
                )
                return sel
            
            # Fall back to UDF-based estimation for complex predicates
            return self._estimate_using_udf(atom, query_spec)

        # traverse PredicateExpr recursively
        def traverse(expr):
            if expr.op == "AND":
                vals = [traverse(arg) for arg in expr.args]
                
                # Use advanced selectivity combination methods
                if self.selectivity_method == "correlated":
                    if self.correlation_matrix is not None:
                        result = self._correlated_and_matrix(vals, self.correlation_matrix)
                    else:
                        result = self._correlated_and_multi(vals, self.correlation_rho)
                elif self.selectivity_method == "blended":
                    result = self._blended_and_multi(vals, self.blending_alpha, self.blending_beta)
                else:  # "naive" or fallback
                    result = np.prod(vals)
                
                if self.debug:
                    method_info = f"({self.selectivity_method})" if self.selectivity_method != "naive" else "(naive)"
                    print(f"  AND{method_info}({', '.join([f'{v:.4f}' for v in vals])}) = {result:.4f}")
                return result
            elif expr.op == "OR":
                vals = [traverse(arg) for arg in expr.args]
                result = 1 - np.prod([1 - v for v in vals])
                if self.debug:
                    print(f"  OR({', '.join([f'{v:.4f}' for v in vals])}) = {result:.4f}")
                return result
            elif expr.op == "ATOM":
                result = handle_atom(expr.atom)
                if self.debug:
                    atom_info = f"{expr.atom.type}"
                    if expr.atom.obj:
                        atom_info += f"(obj={expr.atom.obj}"
                    if expr.atom.other_obj:
                        atom_info += f", other_obj={expr.atom.other_obj}"
                    if expr.atom.value is not None:
                        atom_info += f", value={expr.atom.value}"
                    atom_info += ")" if expr.atom.obj else ""
                    print(f"  ATOM {atom_info} = {result:.4f}")
                return result
            else:
                if self.debug:
                    print(f"  Unknown op '{expr.op}', returning 1.0")
                return 1.0

        selectivity = traverse(preds)
        est_result["selectivity"] = selectivity
        
        if self.debug:
            print(f"\n✓ Final selectivity: {selectivity:.4f}")
        
        return est_result

    def _estimate_using_udf(self, atom: PredicateAtom, query_spec=None) -> float:
        """Estimate selectivity by applying the actual UDF on sampled data.
        
        Args:
            atom: PredicateAtom to estimate selectivity for
            query_spec: Full query specification containing object aliases and classes
            
        Returns:
            Estimated selectivity (fraction of data satisfying the predicate)
        """
        if self.debug:
            print(f"Estimating selectivity for atom: {atom.type}")
        # Check if UDF exists in registry
        all_udfs = self.registry.get_all_udfs()
        if atom.type not in all_udfs:
            # Unknown UDF -> neutral selectivity (no reduction)
            return 1.0
        
        # Get the UDF function from registry
        udf_func = all_udfs[atom.type]
        
        # Get parameter mapping for the UDF
        param_mapping = self.registry.get_udf_param_mapping(atom.type)
        
        # Extract declared classes from query_spec if available
        declared_obj_class = None
        declared_other_obj_class = None
        if query_spec is not None and hasattr(query_spec, 'objects') and hasattr(query_spec.objects, 'aliases'):
            if atom.obj in query_spec.objects.aliases:
                declared_obj_class = query_spec.objects.aliases[atom.obj]['class']
            if atom.other_obj and atom.other_obj in query_spec.objects.aliases:
                declared_other_obj_class = query_spec.objects.aliases[atom.other_obj]['class']
        
        # Map query spec class names to dataset class names
        class_name_mapping = {
            'car': 'vehicle',
            'person': 'pedestrian',
            'bike': 'bicycle',
            'motorcycle': 'motorcycle'
        }
        if declared_obj_class:
            declared_obj_class = class_name_mapping.get(declared_obj_class, declared_obj_class)
        if declared_other_obj_class:
            declared_other_obj_class = class_name_mapping.get(declared_other_obj_class, declared_other_obj_class)
        
        if self.debug and (declared_obj_class or declared_other_obj_class):
            if declared_obj_class:
                print(f"  Declared class for {atom.obj}: {declared_obj_class}")
            if declared_other_obj_class:
                print(f"  Declared class for {atom.other_obj}: {declared_other_obj_class}")
        
        # Sample data for efficiency (use 20% sample)
        sample_df = self.df.sample(frac=0.20, random_state=42)
        
        # Build kwargs by mapping atom attributes to UDF parameters
        # First, identify object parameters
        sig = inspect.signature(udf_func)
        params = list(sig.parameters.keys())
        
        # Determine object parameter names from signature
        obj_params = []
        for param in params:
            if param in ['self']:
                continue
            if 'frame_window' in param:
                continue
            if 'oid' in param.lower() or 'obj' in param.lower() and 'id' in param.lower():
                obj_params.append(param)
        
        # Sample object pairs and evaluate
        results = []
        unique_track_ids = sample_df['track_id'].unique()
        unique_classes = sample_df['class_name'].unique()
        
        # For single-object predicates
        if len(obj_params) == 1:
            for track_id in unique_track_ids:
                obj_data = sample_df[sample_df['track_id'] == track_id]
                if len(obj_data) == 0:
                    continue
                
                # Only consider tracks matching declared class
                if declared_obj_class:
                    obj_class = obj_data['class_name'].iloc[0]
                    if obj_class != declared_obj_class:
                        continue
                
                # Build kwargs
                kwargs = {}
                for param_name, atom_attr in param_mapping.items():
                    atom_val = getattr(atom, atom_attr, None)
                    if atom_val is not None:
                        kwargs[param_name] = atom_val
                
                # Add object parameter
                kwargs[obj_params[0]] = track_id
                
                # Use a frame window covering all frames for this object
                min_frame = obj_data['frame_index'].min()
                max_frame = obj_data['frame_index'].max()
                kwargs['frame_window'] = (min_frame, max_frame)
                
                # Call UDF and accumulate result
                try:
                    result = udf_func(**kwargs)
                    results.append(float(result))
                except Exception as e:
                    # Skip on error
                    continue
        
        # For pairwise predicates
        elif len(obj_params) == 2:
            # Sample object pairs, but only from declared classes if specified
            if declared_obj_class and declared_other_obj_class:
                # Only consider the specific class pair
                class_groups = {}
                if declared_obj_class in unique_classes:
                    class_groups[declared_obj_class] = sample_df[sample_df['class_name'] == declared_obj_class]['track_id'].unique()
                if declared_other_obj_class in unique_classes:
                    class_groups[declared_other_obj_class] = sample_df[sample_df['class_name'] == declared_other_obj_class]['track_id'].unique()
            else:
                # No declared classes, use all classes (backward compatibility)
                class_groups = {cls: sample_df[sample_df['class_name'] == cls]['track_id'].unique() 
                              for cls in unique_classes}
            
            for cls1 in class_groups:
                for cls2 in class_groups:
                    # Check if this class pair matches declared classes
                    if declared_obj_class and declared_other_obj_class:
                        # cls1 should be declared_obj_class and cls2 should be declared_other_obj_class
                        if cls1 != declared_obj_class or cls2 != declared_other_obj_class:
                            continue
                    
                    # Allow same class but ensure different track_ids
                    if cls1 == cls2:
                        if self.debug:
                            print(f"Same class: {cls1}")
                        # Same class: sample different track pairs
                        track_ids = class_groups[cls1]
                        # if self.debug:
                        #     print(f"Track IDs: {track_ids} for class {cls1}")
                        for i, track_id1 in enumerate(track_ids[:20]):  # Limit for efficiency
                            for track_id2 in track_ids[i+1:i+11]:  # Different tracks only
                                if track_id1 == track_id2:
                                    continue
                                
                                # Build kwargs
                                kwargs = {}
                                for param_name, atom_attr in param_mapping.items():
                                    atom_val = getattr(atom, atom_attr, None)
                                    if atom_val is not None:
                                        kwargs[param_name] = atom_val
                                
                                # Add object parameters
                                kwargs[obj_params[0]] = track_id1
                                kwargs[obj_params[1]] = track_id2
                                
                                # Find overlapping frames
                                obj1_data = sample_df[sample_df['track_id'] == track_id1]
                                obj2_data = sample_df[sample_df['track_id'] == track_id2]
                                common_frames = set(obj1_data['frame_index']) & set(obj2_data['frame_index'])
                                
                                if len(common_frames) == 0:
                                    continue
                                
                                min_frame = min(common_frames)
                                max_frame = max(common_frames)
                                kwargs['frame_window'] = (min_frame, max_frame)
                                
                                # Call UDF and accumulate result
                                try:
                                    result = udf_func(**kwargs)
                                    results.append(float(result))
                                except Exception as e:
                                    # Skip on error
                                    continue
                    else:
                        if self.debug:
                            print(f"Different classes: {cls1} and {cls2}")
                        # Different classes: sample from different class combinations
                        for track_id1 in class_groups[cls1][:10]:  # Limit pairs for efficiency
                            for track_id2 in class_groups[cls2][:10]:
                                # if self.debug:
                                #     print(f"class_groups[cls1][:10]: {class_groups[cls1][:10]}")
                                #     print(f"class_groups[cls2][:10]: {class_groups[cls2][:10]}")
                                # Build kwargs
                                kwargs = {}
                                for param_name, atom_attr in param_mapping.items():
                                    atom_val = getattr(atom, atom_attr, None)
                                    if atom_val is not None:
                                        kwargs[param_name] = atom_val
                                
                                # Add object parameters
                                kwargs[obj_params[0]] = track_id1
                                kwargs[obj_params[1]] = track_id2
                                
                                # Find overlapping frames
                                obj1_data = sample_df[sample_df['track_id'] == track_id1]
                                obj2_data = sample_df[sample_df['track_id'] == track_id2]
                                common_frames = set(obj1_data['frame_index']) & set(obj2_data['frame_index'])
                                
                                if len(common_frames) == 0:
                                    continue
                                
                                min_frame = min(common_frames)
                                max_frame = max(common_frames)
                                kwargs['frame_window'] = (min_frame, max_frame)
                                
                                # Call UDF and accumulate result
                                try:
                                    result = udf_func(**kwargs)
                                    results.append(float(result))
                                except Exception as e:
                                    # Skip on error
                                    continue
        
        # Return average selectivity
        if len(results) == 0:
            return 0.0
        return float(np.mean(results))


    @staticmethod
    def _correlated_and_multi(selectivities, rho: float = 0.5) -> float:
        """
        Combine multiple selectivities with correlation adjustment.

        Parameters
        ----------
        selectivities : list[float]
            List of individual predicate selectivities (0–1).
        rho : float
            Average correlation coefficient among predicates.
            0 = fully independent (pure product)
            1 = fully correlated (minimum selectivity dominates)

        Returns
        -------
        float : adjusted combined selectivity
        """
        if not selectivities:
            return 1.0

        # Clamp for stability
        sels = [max(min(s, 1.0), 1e-6) for s in selectivities]
        rho = max(min(rho, 1.0), 0.0)

        # Start from the first
        combined = sels[0]
        for s in sels[1:]:
            combined *= s ** (1 - rho)  # adjust each additional term

        # Optionally, interpolate with max correlation case
        if rho > 0:
            # When perfectly correlated, the smallest selectivity dominates
            correlated_est = min(sels)
            # Weighted blend between correlated_est and adjusted product
            combined = (1 - rho) * combined + rho * correlated_est

        return min(combined, 1.0)


    @staticmethod
    def _correlated_and_matrix(selectivities, rho_matrix):
        """
        Combine multiple selectivities using a pairwise correlation matrix.
        
        Parameters
        ----------
        selectivities : list[float]
            List of individual predicate selectivities (0–1).
        rho_matrix : dict
            Dictionary mapping (i,j) pairs to correlation coefficients.
            Keys are tuples (min(i,j), max(i,j)) for predicates i,j.
            
        Returns
        -------
        float : adjusted combined selectivity
        """
        if not selectivities:
            return 1.0
            
        sels = [max(min(s, 1.0), 1e-6) for s in selectivities]
        combined = 1.0
        
        for i, s in enumerate(sels):
            # average rho for this predicate vs all others
            rho_i = sum(rho_matrix.get((min(i,j), max(i,j)), 0.0) for j in range(len(sels)) if j != i)
            rho_i /= max(len(sels) - 1, 1)
            combined *= s ** (1 - rho_i)
            
        return min(combined, 1.0)


    @staticmethod
    def _blended_and_multi(selectivities, alpha: float = 1.0, beta: float = 3.0) -> float:
        """
        Combine multiple selectivities using log-linear (beta) blending.

        Parameters
        ----------
        selectivities : list[float]
            List of individual predicate selectivities (0–1).
        alpha : float
            Baseline term; usually 1.0
        beta : float
            Scaling term; higher -> less aggressive shrink

        Returns
        -------
        float : adjusted combined selectivity
        """
        if not selectivities:
            return 1.0

        sels = [max(min(s, 1.0), 1e-6) for s in selectivities]
        product = 1.0
        for s in sels:
            product *= s

        blended = product / (alpha + beta * product)
        return min(max(blended, 0.0), 1.0)