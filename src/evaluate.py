# src/evaluate.py
import sys, json, os

if len(sys.argv) < 2:
    print("Usage: python src/evaluate.py <path_to_sources.json>")
    sys.exit(1)

json_path = sys.argv[1]

if not os.path.exists(json_path):
    print(f"File not found: {json_path}")
    sys.exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    sources = json.load(f)

print("Loaded sources:")
for src in sources:
    print("-", src["title"])
