import pickle
from specs import print_spec_details

# === Load the original spec ===
input_path = "spec.pkl"
output_path = "spec_modified.pkl"

spec = pickle.load(open(input_path, "rb"))

print("=== Before modification ===")
print_spec_details(spec)

# === Modify K1 ===
for kf in spec.keyframes:
    if kf.name == "k1" and kf.where.atom.type == "dist_within_two_obj":
        # print(f"Updating {kf.name}: value {kf.where.atom.value} → 80")
        kf.where.atom.obj = "vehicle1"
        kf.where.atom.other_obj = "pedestrian1"
        kf.where.atom.value = 22


# === Modify K2 ===
for kf in spec.keyframes:
    if kf.name == "k2" and kf.where.atom.type == "velocity_below":
        # print(f"Updating {kf.name}: value {kf.where.atom.value} → 0.6")
        kf.where.atom.obj = "vehicle1"
        kf.where.atom.value = 0.8

# === Modify K3 ===
found_k3 = False
for kf in spec.keyframes:
    if kf.name == "k3":
        # print(f"Updating {kf.name}: type {kf.where.atom.type}, value {kf.where.atom.value} → 3.0")
        kf.where.atom.type = "velocity_above"
        kf.where.atom.obj = "vehicle1"
        kf.where.atom.value = 1.8

# === Save the modified spec ===
with open(output_path, "wb") as f:
    pickle.dump(spec, f)

print("\n=== After modification ===")
print_spec_details(spec)
print(f"✅ Modified spec saved to: {output_path}")