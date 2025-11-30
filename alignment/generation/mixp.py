import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--ref", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.policy, "r") as f:
    data_policy = json.load(f)["instances"]

with open(args.ref, "r") as f:
    data_ref = json.load(f)["instances"]

merged_instances = []
for p, r in zip(data_policy, data_ref):
    merged_instances.append({
        "prompt": p["prompt"],
        "responses": p["responses"] + r["responses"]
    })

result = {"type": "text_only", "instances": merged_instances}
with open(args.output, "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✔ Merged data saved to {args.output} | total: {len(merged_instances)}")
