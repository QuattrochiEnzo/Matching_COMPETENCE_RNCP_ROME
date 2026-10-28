import os
import json
import html
import re
import requests
from bs4 import BeautifulSoup

CACHE_PATH = "../Data/RNCP/rncp_cache.json"

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_details_bloc_competence(code_bc, cache=None):

    if cache is None:
        cache = load_cache()

    if code_bc in cache:
        return cache[code_bc]

    match = re.match(r"(RNCP\d+)(BC\d+)", code_bc, re.IGNORECASE)
    if not match:
        raise ValueError("Format attendu : RNCPxxxxxBCxx")

    rncp_code, bc_code = match.groups()
    url = f"https://www.francecompetences.fr/recherche/rncp/{rncp_code[4:]}/"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"❌ Impossible d'accéder à la fiche RNCP {rncp_code}")
    soup = BeautifulSoup(response.text, "html.parser")
    titres_blocs = soup.find_all("p", class_="text--fcpt-certification__title")
    code_bc_clean = code_bc.replace(" ", "").upper()
    for p in titres_blocs:
        texte = html.unescape(p.get_text(strip=True))
        texte_clean = texte.replace(" ", "").upper()

        if code_bc_clean in texte_clean:
            table = p.find_next("table", class_="table--fcpt-certification")
            if not table:
                raise ValueError(f"❌ Tableau introuvable juste après {code_bc}")

            td = table.find("td")
            if not td:
                raise ValueError(f"❌ Aucune cellule trouvée dans le tableau pour {code_bc}")

            items = []

            # 🔹 Cas 1 : <ul><li>
            uls = td.find_all("ul")
            if uls:
                for ul in uls:
                    for li in ul.find_all("li"):
                        txt = html.unescape(li.get_text(strip=True))
                        if txt:
                            items.append(txt)
            else:
                # 🔹 Cas 2 : <p> avec retour à la ligne ou tirets/points médians
                ps = td.find_all("p")
                for p in ps:
                    raw = p.decode_contents()
                    raw = re.sub(r'<br\s*/?>', '\n', raw, flags=re.IGNORECASE)
                    text_blob = BeautifulSoup(raw, "html.parser").get_text(separator="\n", strip=True)

                    for line in text_blob.split("\n"):
                        line = html.unescape(line.strip())

                        if line.endswith(":") and line.count("-") <= 1:
                            continue

                        # Split sur - ou · si non précédé d'une lettre ou chiffre
                        fragments = re.split(r"(?<![a-zA-Z0-9])\s*[-·]\s*", line)
                        for frag in fragments:
                            frag = frag.lstrip("-•●–· ").strip()
                            if len(frag) > 10 and "modalité" not in frag.lower():
                                items.append(frag)

                # 🔹 Cas 3 : fallback brut si on n'a rien trouvé
                if not items:
                    raw_html = td.decode_contents()
                    raw_html = re.sub(r'<br\s*/?>', '\n', raw_html, flags=re.IGNORECASE)
                    text_blob = BeautifulSoup(raw_html, "html.parser").get_text(separator="\n", strip=True)
                    for line in text_blob.split("\n"):
                        line = html.unescape(line.strip())
                        if line.startswith("-") or line.startswith("·"):
                            line = line.lstrip("-•●–· ").strip()
                            if len(line) > 10 and "modalité" not in line.lower():
                                items.append(line)

                # 🔹 Dernier fallback : texte brut, split ligne par ligne
                raw_html = td.decode_contents()
                raw_html = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
                full_text = BeautifulSoup(raw_html, "html.parser").get_text(separator="\n", strip=True)
                lines = full_text.split("\n")
                for line in lines:
                    line = html.unescape(line.strip())

                    if line.endswith(":") and line.count("-") <= 1:
                        continue

                    fragments = re.split(r"(?<![a-zA-Z0-9])\s*-\s*", line)
                    for frag in fragments:
                        # Ensuite on resplit chaque fragment sur le point médian ·
                        subfragments = re.split(r"\s*·\s*", frag)
                        for subfrag in subfragments:
                            subfrag = subfrag.lstrip("-•●–· ").strip()
                            if len(subfrag) > 10 and "modalité" not in subfrag.lower():
                                items.append(subfrag)

            # Nettoyage final : doublons, sous-phrases
            seen = {}
            for phrase in items:
                cleaned = re.sub(r"\s+", " ", phrase.strip())
                short_key = cleaned[:60].lower()
                if short_key not in seen or len(cleaned) > len(seen[short_key]):
                    seen[short_key] = cleaned

            phrases_uniques = list(seen.values())
            filtered = []
            for i, phrase in enumerate(phrases_uniques):
                if not any(
                    phrase != other and phrase in other
                    for other in phrases_uniques
                ):
                    filtered.append(phrase)

            items = filtered

            print("📋 Après nettoyage final :")
            for i, comp in enumerate(items):
                print(f"  {i+1}. {comp}")

            result = {
                "code_bc": code_bc,
                "libelle": texte,
                "competences": items
            }
            cache[code_bc] = result
            save_cache(cache)
            return result

    raise ValueError(f"❌ Bloc {code_bc} introuvable dans la fiche RNCP{rncp_code}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--code_bc", required=True, help="Code du bloc de compétence, ex: RNCP36784BC01")
    args = parser.parse_args()

    try:
        res = get_details_bloc_competence(args.code_bc)
        print(f"\n🔹 Bloc : {res['code_bc']}")
        print(f"🔸 Libellé : {res['libelle']}\n")
        print("📋 Compétences :")
        for c in res["competences"]:
            print("  •", c)
    except ValueError as e:
        print(str(e))
