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

## Garde-fous et limites

- les datasets et checkpoints ne sont pas committés dans Git ;
- les poids du modèle scratch commencent réellement au hasard ;
- les sorties dépendent uniquement de notre corpus et de notre entraînement ;
- le petit modèle n'aura pas les connaissances générales d'un grand LLM ;
- les tests vérifient tokenizer, forward pass, loss, backward pass et génération ;
- une CI dédiée installe PyTorch et teste le moteur scratch séparément ;
- sur Windows avec une carte AMD, l'entraînement GPU PyTorch officiel peut ne pas être disponible directement ; CPU ou environnement Linux/ROCm compatible restent les voies réalistes.

## Licence des données

Le dépôt ne contient aucun livre. Le script Gutenberg télécharge uniquement les identifiants configurés et retire les en-têtes de distribution avant préparation. Avant un entraînement plus large, chaque source devra être vérifiée et documentée séparément.
