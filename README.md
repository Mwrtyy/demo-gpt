# Second Brain

Ce dépôt contient désormais deux moteurs séparés :

1. **Second Brain Web** — l'application existante avec mémoire, feedback et modèle distant.
2. **Second Brain Zero** — notre modèle anglais entraîné depuis des poids aléatoires, sans API d'une autre IA.

## Second Brain Zero — modèle réellement scratch

Le code se trouve dans [`scratch/`](scratch/README.md).

La configuration Level 1 construit un Transformer decoder-only de **19 143 168 paramètres** :

- tokenizer UTF-8 byte-level de 256 tokens, écrit dans le dépôt ;
- contexte de 256 bytes ;
- 6 blocs Transformer ;
- 8 têtes d'attention ;
- largeur interne de 512 ;
- poids initialisés aléatoirement ;
- entraînement next-byte prediction sur un corpus anglais ;
- aucun appel à OpenAI, Anthropic, Google ou une API d'inférence externe.

Installation indépendante de l'application historique :

```bash
python -m venv .venv
# PowerShell : .venv\Scripts\Activate.ps1
python -m pip install -r scratch/requirements.txt
```

Télécharger un premier corpus anglais du domaine public et le préparer :

```bash
python -m scratch.download_gutenberg
python -m scratch.prepare_corpus
```

Vérifier toute la chaîne sur CPU avec le mini-modèle :

```bash
python -m scratch.train \
  --config scratch/configs/smoke_cpu.json \
  --out-dir scratch/checkpoints/smoke
```

Lancer le modèle Level 1 :

```bash
python -m scratch.train \
  --config scratch/configs/level1_english_19m.json \
  --out-dir scratch/checkpoints/level1
```

Générer du texte depuis notre checkpoint :

```bash
python -m scratch.generate \
  --checkpoint scratch/checkpoints/level1/latest.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 400
```

Cette première version apprend à compléter du texte anglais. Elle ne devient un assistant conversationnel qu'après une deuxième phase d'instruction training sur des dialogues propres.

## Tester Second Brain Zero directement dans le site

Le site contient maintenant un laboratoire dédié à l'adresse :

```text
http://localhost:8000/zero
```

Installation locale complète :

```bash
python -m pip install -e ".[dev]"
python -m pip install -r scratch/requirements.txt
second-brain-web
```

Deux méthodes permettent d'activer les poids :

1. copier un checkpoint entraîné vers `runtime/zero/latest.pt` ;
2. ouvrir `/zero`, saisir le jeton administrateur puis uploader le fichier `latest.pt`.

Le laboratoire affiche :

- disponibilité de PyTorch ;
- présence et taille du checkpoint ;
- nombre de paramètres ;
- étape d'entraînement et meilleure validation enregistrée ;
- CPU/GPU utilisé ;
- température, top-k, seed et nombre de nouveaux tokens ;
- vitesse réelle de génération.

Le backend charge les poids localement et appelle directement `ByteGPT.generate`. Aucune requête n'est envoyée à une API de modèle externe.

Pour une image serveur contenant PyTorch CPU :

```bash
docker build -f Dockerfile.zero -t second-brain-zero .
docker run --rm -p 8000:8000 \
  -e SECOND_BRAIN_ADMIN_TOKEN=change-me \
  -v second-brain-zero-data:/data \
  second-brain-zero
```

Le fichier `render-zero.yaml` fournit aussi une configuration de serveur avec disque persistant. Après le déploiement, uploader le checkpoint depuis `/zero`.

## Second Brain Web — application historique

L'application web fournit :

- un chat connecté à un modèle distant ;
- une mémoire SQLite persistante et inspectable ;
- l'historique des interactions ;
- l'ajout manuel de souvenirs ;
- la notation des réponses ;
- un panneau administrateur pour comparer des instructions candidates ;
- deux jetons optionnels pour séparer l'accès normal de l'administration.

Après l'installation du projet principal :

```bash
python -m pip install -e ".[dev]"
second-brain-web
```

Puis ouvrir `http://localhost:8000`.

## Architecture scratch

```text
scratch/
├── tokenizer.py                  # encodage UTF-8 byte-level
├── model.py                      # Transformer decoder-only
├── data.py                       # corpus binaire memory-mapped
├── download_gutenberg.py         # corpus anglais public-domain
├── prepare_corpus.py             # nettoyage et split train/validation
├── train.py                      # optimisation et checkpoints
├── generate.py                   # inférence locale
├── requirements.txt              # aucune dépendance d'API IA
└── configs/
    ├── smoke_cpu.json
    └── level1_english_19m.json
```

```text
src/second_brain/
├── web.py                        # routes normales et routes /api/zero
├── zero_runtime.py               # chargement et inférence du checkpoint
└── static/
    ├── zero.html
    ├── zero.css
    └── zero.js
```

## Garde-fous et limites

- les datasets et checkpoints ne sont pas committés dans Git ;
- les poids du modèle scratch commencent réellement au hasard ;
- les sorties dépendent uniquement de notre corpus et de notre entraînement ;
- le petit modèle n'aura pas les connaissances générales d'un grand LLM ;
- l'upload d'un checkpoint exige le jeton administrateur ;
- le fichier est chargé et validé avant de remplacer le checkpoint actif ;
- le chargeur PyTorch utilise le mode restreint `weights_only=True` ;
- les tests vérifient tokenizer, forward pass, loss, backward pass, checkpoint et génération ;
- une CI dédiée installe PyTorch et teste le moteur scratch séparément ;
- sur Windows avec une carte AMD, l'entraînement GPU PyTorch officiel peut ne pas être disponible directement ; CPU ou environnement Linux/ROCm compatible restent les voies réalistes.

## Licence des données

Le dépôt ne contient aucun livre. Le script Gutenberg télécharge uniquement les identifiants configurés et retire les en-têtes de distribution avant préparation. Avant un entraînement plus large, chaque source devra être vérifiée et documentée séparément.
