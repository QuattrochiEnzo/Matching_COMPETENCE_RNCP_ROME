import pandas as pd
import sys

if len(sys.argv) != 3:
    print("Usage: python3 graphe_csv_to_tlp.py input.csv output.tlp")
    sys.exit(1)

csv_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv(csv_path)

# Récupérer tous les noms de noeuds uniques
nodes = set(df["source"]).union(set(df["target"]))
node_to_id = {name: str(i) for i, name in enumerate(sorted(nodes))}

with open(output_path, "w") as f:
    f.write("(tlp \"2.0\"\n")
    f.write("(nodes " + " ".join(node_to_id.values()) + ")\n")

    for i, row in df.iterrows():
        src = node_to_id[row["source"]]
        tgt = node_to_id[row["target"]]
        eid = f"{src}_{tgt}"
        f.write(f"(edge {eid} {src} {tgt})\n")

    f.write("(property 0 string \"viewLabel\"\n")
    f.write("(default \"\" \"\")\n")
    f.write("(node\n")
    for name, nid in node_to_id.items():
        label = name.replace('"', "'")
        f.write(f"({nid} \"{label}\")\n")
    f.write(")\n")  # fin node label

    f.write("(property 0 float \"weight\"\n")
    f.write("(default \"1\" )\n")
    f.write("(edge\n")
    for i, row in df.iterrows():
        src = node_to_id[row["source"]]
        tgt = node_to_id[row["target"]]
        eid = f"{src}_{tgt}"
        f.write(f"({eid} {row['weight']})\n")
    f.write(")\n")  # fin edge weight

    f.write(")\n")  # fin tlp
