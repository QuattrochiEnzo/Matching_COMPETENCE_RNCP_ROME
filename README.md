Voici le **README complet avec la partie mise à jour** :

````markdown
# 🔗 Lien RNCP–ROME : Test de similarité avec embeddings ou TF-IDF

Ce projet permet d’expérimenter la mise en relation entre compétences RNCP et fiches métiers ROME à l’aide :
- de modèles d’embeddings (`sentence-transformers`)
- ou d’une approche TF-IDF avec lemmatisation (`stanza`).

---

## ⚙️ Installation

### 1. Cloner ou copier ce dépôt
Placez-vous dans le dossier voulu :

```bash
git clone git@gitlab.emi.u-bordeaux.fr:equattrochi/labri-stage-modelisation_rncp_rome.git rncp-rome-link
cd rncp-rome-link
````

### 2. Créer un environnement Python

```bash
python3 -m venv env_poke
source env_poke/bin/activate
```

Sur Windows (CMD) :

```cmd
env_poke\Scripts\activate
```

### 3. Installer les dépendances

#### Option recommandée (CPU uniquement, plus léger)

```bash
pip install --upgrade pip
pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install sentence-transformers Pillow
pip install transformers==4.52.4 pandas scikit-learn numpy==1.26.4 tqdm stanza
```

---

> ❗ **Si `torch` a installé CUDA par erreur** :
>
> ```bash
> pip uninstall torch sentence-transformers
> pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
> pip install sentence-transformers --no-deps
> pip install Pillow
> ```

---

> 💡 **Vérification CPU uniquement**
>
> ```python
> import torch
> print(torch.__version__)
> print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
> ```

---

## 🚀 Lancer les tests

### 📁 Se placer dans le dossier `src`

```bash
cd src
```

### 🧠 Comparaison embeddings sur plusieurs fiches RNCP–ROME (avec logs)

```bash
python poke_embedding.py
```

### 🔍 Détail embeddings pour un couple RNCP–ROME

```bash
python poke_embedding_detail.py --rncp RNCP34067 --rome H1402 --seuil 0.35
```

### 📊 Comparaison TF-IDF multi RNCP–ROME (avec logs)

```bash
python poke_tfidf.py
```

### 🧪 Détail TF-IDF pour un couple précis

```bash
python poke_tfidf_detail.py --rncp RNCP34067 --rome H1402 --seuil 0.15
```

---

## 🗃 Contenu du projet

* `src/poke_embedding.py` : comparaison embeddings multiple (résumé + logs)
* `src/poke_embedding_detail.py` : comparaison embeddings détaillée (1 couple)
* `src/poke_tfidf.py` : comparaison TF-IDF multiple (résumé + logs)
* `src/poke_tfidf_detail.py` : comparaison TF-IDF détaillée (1 couple)
* `logs/` : résultats enregistrés au format CSV pour la pluspart
* `README.md` : ce fichier

---

## 🆕 Autres modules et fonctionnalités

### 🔄 Parsing & Données d’entrée

* `GofParser.py` / `gofParser2.py` / `gofParser3.py` : Extraction des compétences depuis des documents GOF (offres de formation).
* `enrichirGof.py` : Enrichissement des données GOF avant analyse.
* `grapheOffre.py` : Génération de graphes de relations entre offres et métiers.

### 📊 Analyse de clusters & Tests exploratoires

* `explore_k_rome.py` : Analyse des clusters ROME pour déterminer leur cohérence.
* `compare_clusters.py` : Comparaison de regroupements de fiches ROME.

### 🧩 Matching GOF / ROME

* `gofEmbedding.py` : Similarité GOF–ROME par embeddings.
* `offre_sim.py` / `offre_sim_rome.py` : Recherche d'offres ou métiers proches d’une entrée.

### 🌐 Pages Dash (Interface Web)

* `pages/RncpToRncp.py` : Pas fonctionnel le cacul des 40 milles fiches est trop long, il faudrait trouver une alternative de calcul.
* `pages/RomeToRome.py` : Recherche des 5 fiches ROME les plus proches.
* `pages/RomeToRomePrecision.py` : Comparaison détaillée entre deux fiches ROME.
* `pages/RncpToRome.py` *(modifié)* : Comparaison classique RNCP–ROME.
* `pages/comparaison.py` / `pages/embedding.py` / `pages/tfidf.py` / `pages/home.py` *(modifiés)* : Pages principales de l’application.

### 🕸 Visualisation & Graphes Tulip

* `TulipEvering.py` / `TulipRome.py` / `TulipScript.py` / `GrapheEvering.py` : Génération de graphes Tulip (visualisation des liens).

### 🖥️ Application principale

* `app.py` *(modifié)* : Point d’entrée Dash multi-pages.

---

## 🧽 Nettoyage

Pour désactiver l’environnement :

```bash
deactivate
```

---

