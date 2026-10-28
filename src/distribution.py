import pandas as pd
import ast
from collections import Counter
import argparse
import re

def load_log_file(code_rome: str, code_rncp: str, method: str, reverse: bool) -> pd.DataFrame:
    assert method in ["embedding", "tfidf"], "Méthode inconnue : embedding ou tfidf uniquement"
    
    log_filename = f"../logs/logs_{method}{'_reverse' if reverse else ''}__{code_rncp.upper()}__{code_rome.upper()}.csv"
    print(f"🔍 Chargement : {log_filename}")
    
    try:
        return pd.read_csv(log_filename)
    except FileNotFoundError:
        print("❌ Fichier introuvable.")
        return None

def extract_competences(df: pd.DataFrame, top_col: str = "Top_3") -> Counter:
    all_competences = []

    for row in df[top_col].dropna():
        # Séparer la chaîne sur les lignes (si plusieurs compétences sont dans une seule cellule)
        lignes = str(row).split("\\n") if "\\n" in row else str(row).split("\n")
        for ligne in lignes:
            # Supprimer les scores type "— score = 0.7231" ou " (score = 0.7231)"
            ligne = re.sub(r"(—|--)?\s*\(?score\s*=\s*[0-9.,]+\)?", "", ligne, flags=re.IGNORECASE)
            
            # Supprimer les parenthèses vides ou résiduelles
            ligne = re.sub(r"\(\s*\)", "", ligne)
            
            texte = ligne.strip().lower()
            if texte:
                all_competences.append(texte)


    return Counter(all_competences)

def display_top(counter: Counter, n: int = 20):
    print("\n📈 Top compétences matchées :\n")

    for competence, freq in counter.most_common(n):
        print("🧠 Compétence :", competence.strip())
        print(f"🔢 Nombre d’occurrences : {freq}\n")


# === Exemple d’appel ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse des compétences RNCP matchées dans les logs")
    parser.add_argument("--rncp", type=str, required=True)
    parser.add_argument("--rome", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["embedding", "tfidf"], required=True)
    parser.add_argument("--reverse", type=lambda x: x.lower() == "true", default=False)

    args = parser.parse_args()

    df = load_log_file(args.rome, args.rncp, args.method, args.reverse)
    if df is not None:
        counter = extract_competences(df, top_col="top_3_associees")
        display_top(counter)