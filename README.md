# Second Brain

Second Brain est un agent IA **auto-améliorant mais contrôlé**. Il ne prétend pas réentraîner magiquement son modèle de base. Il améliore ce qui est réellement modifiable par une application :

- sa mémoire personnelle ;
- ses instructions actives ;
- sa stratégie de réponse ;
- son benchmark ;
- plus tard, son code, uniquement dans une branche Git et après tests.

## Ce que signifie « s'auto-améliorer »

Un cycle d'amélioration exécute cinq étapes :

1. évaluer la version active sur un jeu de tests stable ;
2. analyser les échecs sans copier artificiellement les réponses attendues ;
3. générer une nouvelle version candidate des instructions ;
4. réévaluer cette candidate sur exactement le même benchmark ;
5. promouvoir la candidate seulement si le score progresse et qu'aucun cas critique ne régresse.

Par défaut, `improve` **ne remplace rien**. Il écrit une candidate et un rapport dans `state/`. L'option `--auto-promote` ne fonctionne que si la barrière de promotion est validée.

## Architecture

```text
src/second_brain/
├── core.py          # boucle de réponse
├── memory.py        # mémoire SQLite inspectable
├── evaluation.py    # benchmark et scores déterministes
├── improvement.py   # génération, comparaison et barrière de promotion
├── prompt_store.py  # versions, candidates, archive et promotion
├── llm.py           # adaptateur OpenAI Responses API
└── cli.py           # commandes locales

prompts/active.json  # comportement actif versionné
data/evals.jsonl     # benchmark stable
state/               # candidates et rapports, ignorés par Git
```

## Installation

```bash
git clone https://github.com/Mwrtyy/demo-gpt.git
cd demo-gpt
git switch agent/self-improving-core
python -m venv .venv
```

Activation sous PowerShell :

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
$env:OPENAI_API_KEY="sk-..."
```

## Utilisation

```bash
second-brain status
second-brain ask "Explique-moi les probabilités simplement"
second-brain remember "Je préfère des explications directes en français"
second-brain feedback 1 0.9 --note "Réponse claire"
second-brain eval
second-brain improve
second-brain improve --auto-promote
second-brain promote state/candidates/candidate-YYYYMMDDTHHMMSSZ.json
```

## Garde-fous

- aucune clé API n'est enregistrée dans Git ;
- aucune candidate n'écrase directement le prompt actif ;
- une amélioration moyenne ne peut pas masquer une régression critique ;
- les tests GitHub Actions n'appellent pas l'API et ne dépensent aucun crédit ;
- la mémoire est locale, explicite et supprimable ;
- V1 ne donne pas à l'agent un shell autonome ni le droit de fusionner son propre code.

## Limite honnête

Cette V1 améliore le **système autour du modèle**, pas les poids internes de GPT. Une V2 pourra générer des patches de code dans un bac à sable, lancer les tests, ouvrir automatiquement une pull request et attendre une validation humaine. Une V3 pourra enrichir le benchmark à partir des retours négatifs, avec déduplication et validation pour empêcher l'agent de rendre ses propres tests artificiellement faciles.
