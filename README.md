# Second Brain

Ce dépôt contient deux moteurs séparés :

1. **Second Brain Web** — application avec mémoire, feedback et modèle distant.
2. **Second Brain Zero** — notre Transformer anglais entraîné depuis des poids aléatoires, sans API d'une autre IA.

## Second Brain Zero — modèle réellement scratch

Le code se trouve dans [`scratch/`](scratch/README.md). La configuration Level 1 construit un Transformer decoder-only de **19 143 168 paramètres** :

- tokenizer UTF-8 byte-level de 256 tokens écrit dans le dépôt ;
- contexte de 256 bytes ;
- 6 blocs Transformer ;
- 8 têtes d'attention ;
- largeur interne de 512 ;
- poids initialisés aléatoirement ;
- entraînement next-byte prediction sur un corpus anglais ;
- aucun appel à OpenAI, Anthropic, Google ou une API d'inférence externe.

Installation locale :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip install -r scratch/requirements.txt
```

Lancer le serveur avec un jeton administrateur :

```powershell
$env:SECOND_BRAIN_ADMIN_TOKEN="change-me"
python -m uvicorn second_brain.web:app --host 127.0.0.1 --port 8000
```

Puis ouvrir :

```text
http://127.0.0.1:8000/zero
```

## Entraînement autonome depuis le site

La page `/zero` contient un **Training Orchestrator**. Après avoir enregistré le même jeton administrateur que celui configuré dans PowerShell, un seul clic peut :

- télécharger et préparer le corpus quand il manque ;
- démarrer une nouvelle génération ou reprendre `latest.pt` ;
- afficher étape, loss, validation, vitesse, ETA et logs ;
- demander une pause sûre après l'étape courante ;
- reprendre ou arrêter sans corrompre le checkpoint ;
- conserver `best.pt` séparément de `latest.pt` ;
- activer automatiquement le meilleur checkpoint dans le laboratoire ;
- reprendre après le redémarrage du serveur grâce à l'état persistant ;
- passer à la génération suivante uniquement si le seuil de validation et la limite de paramètres sont respectés.

Les générations approuvées sont définies dans `scratch/configs/growth_plan.json` :

```text
Smoke   67 776 paramètres
Level 1 19 143 168 paramètres
Level 2 38 023 680 paramètres
```

La croissance n'a pas lieu au milieu d'un entraînement. La génération suivante reçoit les tenseurs compatibles de la meilleure génération précédente ; les nouvelles couches commencent aléatoirement.

### Important pour un entraînement déjà lancé manuellement

Le tableau de bord ne peut pas prendre le contrôle d'un processus `python -m scratch.train` déjà lancé dans un autre PowerShell. Arrêter d'abord ce processus avec `Ctrl+C`, mettre le dépôt à jour, relancer le serveur, puis cliquer sur **Start / resume training**. Le tableau de bord reprendra automatiquement le fichier `scratch/checkpoints/<generation>/latest.pt`.

### Garde-fous

- les commandes de lancement, pause, reprise et arrêt exigent le jeton administrateur ;
- la pause et l'arrêt prennent effet à la frontière d'une étape d'optimisation ;
- le meilleur checkpoint n'est promu qu'après une amélioration de validation ;
- la croissance automatique est désactivée par défaut ;
- une limite de paramètres bloque les générations trop grandes ;
- le système ne télécharge pas arbitrairement du texte sur tout Internet ;
- les datasets et checkpoints restent exclus de Git.

## Entraînement manuel

Télécharger et préparer le corpus :

```powershell
python -m scratch.download_gutenberg
python -m scratch.prepare_corpus
```

Mini-modèle de vérification :

```powershell
python -m scratch.train `
  --config scratch/configs/smoke_cpu.json `
  --out-dir scratch/checkpoints/smoke
```

Level 1 :

```powershell
python -m scratch.train `
  --config scratch/configs/level1_english_19m.json `
  --out-dir scratch/checkpoints/level1
```

Reprendre :

```powershell
python -m scratch.train `
  --config scratch/configs/level1_english_19m.json `
  --out-dir scratch/checkpoints/level1 `
  --resume scratch/checkpoints/level1/latest.pt
```

Générer directement en ligne de commande :

```powershell
python -m scratch.generate `
  --checkpoint scratch/checkpoints/level1/latest.pt `
  --prompt "Once upon a time" `
  --max-new-tokens 400
```

Cette première version apprend à compléter du texte anglais. Elle ne devient un assistant conversationnel qu'après une phase ultérieure d'instruction training sur des dialogues propres.

## Inférence locale dans le navigateur

Le laboratoire affiche :

- disponibilité de PyTorch ;
- présence et taille du checkpoint actif ;
- nombre de paramètres ;
- étape d'entraînement et meilleure validation ;
- CPU/GPU utilisé ;
- température, top-k, seed et nombre de nouveaux tokens ;
- vitesse réelle de génération.

Le backend charge les poids localement et appelle directement `ByteGPT.generate`. Aucune requête n'est envoyée à une API de modèle externe.

Deux méthodes permettent d'activer manuellement les poids :

1. copier un checkpoint vers `runtime/zero/latest.pt` ;
2. saisir le jeton administrateur dans `/zero` et uploader un fichier `.pt`.

## Image serveur CPU

```bash
docker build -f Dockerfile.zero -t second-brain-zero .
docker run --rm -p 8000:8000 \
  -e SECOND_BRAIN_ADMIN_TOKEN=change-me \
  -v second-brain-zero-data:/data \
  second-brain-zero
```

Le fichier `render-zero.yaml` fournit une configuration avec disque persistant.

## Architecture

```text
scratch/
├── tokenizer.py
├── model.py
├── data.py
├── download_gutenberg.py
├── prepare_corpus.py
├── train.py                       # événements, pause, reprise, best.pt
├── generate.py
├── requirements.txt
└── configs/
    ├── smoke_cpu.json
    ├── level1_english_19m.json
    ├── level2_english_38m.json
    └── growth_plan.json
```

```text
src/second_brain/
├── web.py                         # API web et contrôles administrateur
├── zero_runtime.py                # chargement et inférence locale
├── training_orchestrator.py       # processus, état, reprise et croissance
└── static/
    ├── zero.html
    ├── zero.css
    └── zero.js
```

## Limites matérielles

Le modèle Level 1 est déjà beaucoup plus lent que le smoke model sur CPU. Level 2 double presque la taille de Level 1 et peut demander plusieurs jours ou semaines selon la machine. Sur Windows avec une carte AMD, PyTorch peut continuer à utiliser le CPU ; Linux/ROCm compatible ou un GPU cloud devient la voie réaliste pour aller plus loin.

## Licence des données

Le dépôt ne contient aucun livre. Le script Gutenberg télécharge uniquement les identifiants configurés et retire les en-têtes de distribution avant préparation. Avant un entraînement plus large, chaque source doit être vérifiée et documentée séparément.
