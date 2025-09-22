# FAST_VEL = 2.0
# STRAIGHT_TOL = 5.0
# PERP_DEG = 90.0
# DUR_SEC = 3.0

# SCHEMA = {
#     "entities": {
#         "car": {"props": {"id": "int"}}
#     },
#     "ops_single_frame": {
#         "velocity_above": ["Entity", "float"],
#         "velocity_below": ["Entity", "float"],
#         "heading_diff_to": ["Entity", "Entity", "deg", "tol_deg?float"],
#         "distance_within": ["Entity", "float"],
#         "distance_apart": ["Entity", "float"],
#     },
#     "ops_inter_frame": {
#         "heading_diff": ["Entity", "deg", "tol_deg?float"],
#     },
#     "temporal": {
#         "always": ["target:KF", "dur_sec:float", "tol?float", "anchor?None|KF"],
#         "interframe": ["anchor:KF", "target:KF", "time_shift:float", "comparators:list"],
#     },
# }