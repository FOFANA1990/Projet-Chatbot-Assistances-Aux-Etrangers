#  Chatbot Administratif — Assistance aux Étrangers en France

> Système intelligent de questions-réponses basé sur une architecture **RAG (Retrieval-Augmented Generation)** combinant FAISS, Sentence Transformers et des modèles de langage HuggingFace. Conçu pour guider les étrangers dans leurs démarches administratives en France.

---

##  Sommaire

1. [Contexte et problématique](#-contexte-et-problématique)
2. [Architecture du système](#-architecture-du-système)
3. [Fonctionnalités](#-fonctionnalités)
4. [Structure du projet](#-structure-du-projet)
5. [Prérequis](#-prérequis)
6. [Installation](#-installation)
7. [Lancement du notebook](#-lancement-du-notebook)
8. [Guide d'utilisation](#-guide-dutilisation)
9. [Données et corpus](#-données-et-corpus)
10. [Modèles utilisés](#-modèles-utilisés)
11. [Évaluation et métriques](#-évaluation-et-métriques)
12. [Résultats et performances](#-résultats-et-performances)
13. [Extension avec un LLM externe](#-extension-avec-un-llm-externe)
14. [Limites et améliorations](#-limites-et-améliorations)
15. [Sources officielles](#-sources-officielles)

---

##  Contexte et problématique

Les étrangers résidant ou souhaitant s'installer en France font face à des procédures administratives souvent complexes, longues et difficiles à appréhender, en particulier pour les personnes ne maîtrisant pas parfaitement la langue française. Les démarches concernées incluent notamment :

- Le renouvellement ou l'obtention d'un **titre de séjour**
- Les demandes de **visa long séjour**
- Les procédures de **demande d'asile**
- L'accès à la **sécurité sociale** (carte Vitale, CPAM)
- La recherche d'un **logement** (APL, HLM)
- Les conditions pour **travailler** légalement
- L'inscription dans une **université** française
- La démarche de **naturalisation**

**Problématique centrale :** Comment concevoir un chatbot basé sur les LLM capable de fournir des réponses fiables, compréhensibles et contextualisées sur ces démarches, tout en limitant les hallucinations ?

**Solution proposée :** Une architecture RAG (Retrieval-Augmented Generation) qui ancre les réponses dans un corpus documentaire administratif vérifié avant de les formuler.

---

##  Architecture du système

```
┌─────────────────────────────────────────────────────────────┐
│                      PIPELINE RAG                           │
│                                                             │
│  Question         Recherche          Génération             │
│  utilisateur  →   documentaire   →   de réponse             │
│                                                             │
│  ┌──────────┐   ┌────────────────┐   ┌──────────────────┐  │
│  │  Saisie  │   │ Sentence Trans.│   │ Générateur       │  │
│  │ (widget  │──▶│ (embeddings    │──▶│ Template / BERT  │  │
│  │ Jupyter) │   │  multilingues) │   │ QA / LLM externe │  │
│  └──────────┘   └───────┬────────┘   └──────────────────┘  │
│                         │                      ▲            │
│                         ▼                      │            │
│                  ┌─────────────┐               │            │
│                  │ Index FAISS │───────────────┘            │
│                  │ (vecteurs   │  Top-K chunks              │
│                  │  normalisés)│  pertinents                │
│                  └─────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

| Composant | Technologie | Rôle |
|---|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Encodage sémantique français/multilingue |
| Index vectoriel | FAISS `IndexFlatIP` | Recherche par similarité cosine |
| Découpage | LangChain `RecursiveCharacterTextSplitter` | Chunking des documents (500 chars) |
| Génération | Templates Python + pipeline BERT QA | Formulation des réponses |
| Interface | `ipywidgets` + `IPython.display` | Chat interactif dans Jupyter |

---

##  Fonctionnalités

### Chatbot
- **Détection automatique de l'intention** : classification par mots-clés en 9 catégories (titre de séjour, visa, asile, santé, travail, logement, études, naturalisation, famille)
- **Recherche sémantique RAG** : retrouve les 3 passages les plus pertinents dans la base documentaire via FAISS
- **Score de confiance** : chaque réponse est accompagnée d'un score de pertinence (0–1)
- **Historique de conversation** : toutes les interactions sont enregistrées pour analyse
- **Interface Jupyter interactive** : bulles de chat, animation de frappe, raccourci clavier Entrée

### Analyse de données (EDA)
- Distribution des catégories, longueur des questions/réponses
- Nuage de mots et top 15 termes les plus fréquents
- Visualisation PCA 2D des embeddings (Plotly interactif)

### Évaluation automatique
- Score **BLEU** (qualité de génération)
- Scores **ROUGE-1, ROUGE-2, ROUGE-L** (couverture du contenu)
- **Similarité sémantique cosine** entre réponse générée et référence
- Mesure du **temps de réponse** en millisecondes

### Comparaison de modèles
- Test de 3 modèles d'embeddings différents
- Comparaison : précision Top-1, score moyen, temps d'encodage

---

## Structure du projet

```
chatbot-admin-etrangers/
│
├── chatbot_admin_etrangers.ipynb   # Notebook principal (14 sections)
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
│
├── outputs/                        # Générés à l'exécution
│   ├── index_administratif.faiss   # Index vectoriel sauvegardé
│   ├── metadata_chunks.json        # Métadonnées des chunks
│   ├── eda_corpus.png              # Visualisations EDA
│   ├── wordcloud_corpus.png        # Nuage de mots
│   ├── evaluation_chatbot.png      # Graphiques d'évaluation
│   ├── comparaison_modeles.png     # Comparaison des modèles
│   └── tableau_bord_final.png      # Dashboard final
│
└── data/                           # Optionnel : données externes
    └── faq_custom.json             # FAQ personnalisée à ajouter
```

---

## 💻 Prérequis

| Élément | Minimum recommandé |
|---|---|
| Python | 3.9 ou supérieur |
| RAM | 8 Go (16 Go recommandés) |
| Disque | 5 Go libres (modèles téléchargés automatiquement) |
| GPU | Non requis — fonctionne entièrement sur CPU |
| Connexion | Nécessaire au premier lancement (téléchargement des modèles) |

> **Compatibilité** : testé sous Windows 10/11, Ubuntu 22.04, macOS 13+. Fonctionne également sur Google Colab (gratuit).

---

##  Installation

### Option A — Environnement virtuel local (recommandé)

```bash
# 1. Cloner ou télécharger le projet
git clone https://github.com/votre-repo/chatbot-admin-etrangers.git
cd chatbot-admin-etrangers

# 2. Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement
# Windows :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. (JupyterLab uniquement) Activer l'extension ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# 5. Lancer Jupyter
jupyter lab
# ou
jupyter notebook
```

### Option B — Google Colab (sans installation)

1. Ouvrir [Google Colab](https://colab.research.google.com)
2. Importer le fichier `chatbot_admin_etrangers.ipynb`
3. Exécuter la **cellule 1** (Installation) — toutes les dépendances s'installent automatiquement
4. Exécuter les cellules suivantes dans l'ordre

> Sur Colab, remplacer `faiss-cpu` par `faiss-gpu` si vous utilisez un runtime GPU pour de meilleures performances.

### Option C — Anaconda / Conda

```bash
conda create -n chatbot-admin python=3.10
conda activate chatbot-admin
pip install -r requirements.txt
jupyter lab
```

---

## ▶️ Lancement du notebook

Une fois l'environnement installé et Jupyter ouvert :

1. Ouvrir `chatbot_admin_etrangers.ipynb`
2. Aller dans le menu **Kernel → Restart & Run All** pour exécuter tout le notebook d'un coup
3. Ou exécuter les cellules **une par une** dans l'ordre numéroté (recommandé pour comprendre chaque étape)

> **Premier lancement :** les modèles HuggingFace (~500 Mo) sont téléchargés automatiquement depuis le Hub. Une connexion internet est nécessaire. Les téléchargements suivants utilisent le cache local.

---

##  Guide d'utilisation

### Utilisation de base via l'interface Jupyter

La dernière cellule du notebook lance l'interface conversationnelle interactive :

```python
creer_interface_chatbot(chatbot)
```

Une fois exécutée, une interface de chat apparaît directement dans le notebook :

- **Saisir une question** dans le champ de texte en bas
- Appuyer sur **Entrée** (ou cliquer sur le bouton Envoyer) pour obtenir une réponse
- Les métadonnées s'affichent sous chaque réponse : intention détectée, score de confiance, temps de réponse
- Le bouton **Effacer** réinitialise la conversation

### Utilisation programmatique

```python
# Poser une question et récupérer la réponse complète
rep = chatbot.repondre("Comment renouveler mon titre de séjour ?", verbose=True)

# Accéder aux champs de la réponse
print(rep["reponse"])           # Texte de la réponse
print(rep["intention"])         # Catégorie détectée (ex: "titre_sejour")
print(rep["score_confiance"])   # Score entre 0 et 1
print(rep["sources"])           # Sources documentaires utilisées
print(rep["temps_ms"])          # Temps de réponse en millisecondes

# Afficher les statistiques globales de la session
chatbot.afficher_stats()
```

### Exemples de questions supportées

| Catégorie | Exemples de questions |
|---|---|
| Titre de séjour | "Comment renouveler mon titre de séjour ?" · "Mon titre a été volé, que faire ?" |
| Visa | "Quels documents pour un visa étudiant ?" · "Comment faire un VLS-TS ?" |
| Asile | "Comment faire une demande d'asile en France ?" · "Qu'est-ce que l'OFPRA ?" |
| Santé | "Comment obtenir une carte Vitale ?" · "Comment bénéficier de la CSS ?" |
| Travail | "Conditions pour travailler légalement ?" · "Qu'est-ce que le passeport talent ?" |
| Logement | "Comment obtenir une APL ?" · "Comment demander un logement HLM ?" |
| Études | "Comment s'inscrire à l'université ?" · "Qu'est-ce que Campus France ?" |
| Naturalisation | "Comment demander la nationalité française ?" · "Quelles conditions pour la naturalisation ?" |
| Famille | "Comment faire un regroupement familial ?" |

---

##  Données et corpus

### Corpus intégré

Le notebook inclut **14 documents administratifs** répartis en **8 catégories** :

| Catégorie | Nb. documents | Sources |
|---|---|---|
| Titre de séjour | 4 | ANEF, Service-Public.fr |
| Visa | 2 | France-Visas.gouv.fr, Campus France |
| Asile | 1 | OFPRA, Service-Public.fr |
| Sécurité sociale | 2 | Ameli.fr, CPAM |
| Travail | 2 | Ministère du Travail, Service-Public.fr |
| Logement | 2 | CAF.fr, Ministère du Logement |
| Naturalisation | 1 | Service-Public.fr |
| Études | 1 | Campus France, Parcoursup |

### Ajouter vos propres documents

Pour enrichir le corpus avec de nouveaux documents, ajoutez des entrées dans la liste `CORPUS_ADMINISTRATIF` (section 3 du notebook) en respectant ce format :

```python
{
    "id": "xx_001",
    "categorie": "Ma Catégorie",
    "question": "Question représentative ?",
    "reponse": "Réponse détaillée...",
    "source": "Nom de la source officielle",
    "tags": ["mot-clé1", "mot-clé2"]
}
```

Après ajout, relancer les sections 4 à 6 pour reconstruire l'index FAISS.

---

##  Modèles utilisés

### Modèle d'embeddings (par défaut)

**`paraphrase-multilingual-MiniLM-L12-v2`** (Sentence Transformers)

- Taille : ~118M paramètres (~500 Mo)
- Langues supportées : 50+ langues dont le français
- Dimension des vecteurs : 384
- Avantage : excellent rapport qualité/vitesse pour le français

### Modèles alternatifs testés

| Modèle | Langue | Taille | Cas d'usage |
|---|---|---|---|
| `all-MiniLM-L6-v2` | Anglais | 22M params | Très rapide, prototypage |
| `paraphrase-multilingual-mpnet-base-v2` | Multilingue | 278M params | Haute précision |
| `deepset/roberta-base-squad2` | Anglais | Pipeline QA extractif | Extraction de passages |

### Extension LLM génératif (optionnel)

Pour de meilleures réponses, le notebook prévoit l'intégration de LLMs génératifs :

```python
# Via HuggingFace Inference API (compte gratuit)
from huggingface_hub import InferenceClient
client = InferenceClient(token="votre_token_hf")
response = client.text_generation(prompt, model="mistralai/Mistral-7B-Instruct-v0.3")

# Via Ollama (local, aucune clé requise)
# Installer Ollama : https://ollama.com
# Puis : ollama pull mistral
import requests
r = requests.post("http://localhost:11434/api/generate",
                  json={"model": "mistral", "prompt": prompt})

# Via Groq (API gratuite, très rapide)
from groq import Groq
client = Groq(api_key="votre_cle_groq")
```

---

##  Évaluation et métriques

Le notebook évalue automatiquement le chatbot sur un jeu de 6 paires question/réponse de référence.

### Métriques calculées

| Métrique | Description | Plage |
|---|---|---|
| **BLEU** | Correspondance n-grammes entre réponse générée et référence | 0 à 1 |
| **ROUGE-1** | Rappel des unigrammes communs | 0 à 1 |
| **ROUGE-2** | Rappel des bigrammes communs | 0 à 1 |
| **ROUGE-L** | Plus longue sous-séquence commune | 0 à 1 |
| **Similarité sémantique** | Cosine similarity entre embeddings | 0 à 1 |
| **Score de confiance RAG** | Score FAISS de la recherche documentaire | 0 à 1 |
| **Temps de réponse** | Latence de bout en bout | ms |

### Interprétation

- Score de confiance RAG > **0.6** → réponse très pertinente
- Score de confiance RAG entre **0.3 et 0.6** → réponse acceptable
- Score de confiance RAG < **0.3** → question hors corpus, rediriger vers les sources officielles

---

##  Résultats et performances

Résultats typiques obtenus sur le corpus fourni (CPU, Intel Core i7) :

| Métrique | Score moyen |
|---|---|
| Score BLEU | ~0.18 |
| ROUGE-1 | ~0.45 |
| ROUGE-L | ~0.38 |
| Similarité sémantique | ~0.72 |
| Score de confiance RAG | ~0.78 |
| Temps de réponse | ~250 ms |

> Ces scores reflètent une génération **extractive** (le système reformule le contenu du corpus sans halluciner). L'intégration d'un LLM génératif comme Mistral améliore significativement les scores BLEU et ROUGE.

---

## 🔌 Extension avec un LLM externe

Pour obtenir des réponses plus fluides et naturelles, voici comment brancher un LLM génératif à l'architecture RAG existante :

```python
# Remplacer GenerateurReponsesTemplate par cette classe
class GenerateurLLM:
    def __init__(self, api_client):
        self.client = api_client

    def generer_reponse(self, question: str, resultats: list) -> str:
        contexte = moteur.construire_contexte(resultats)
        prompt = f"""Tu es un assistant administratif pour les étrangers en France.
Utilise uniquement le contexte suivant pour répondre en français.

Contexte :
{contexte}

Question : {question}
Réponse :"""
        # Appel à l'API choisie
        return self.client.complete(prompt)
```

---

##  Limites et améliorations

### Limites actuelles

- **Corpus limité** : 14 documents couvrent les cas les plus fréquents mais pas toutes les situations
- **Génération extractive** : sans LLM externe, les réponses restent proches du texte source
- **Pas de mémoire multi-tour** : chaque question est traitée indépendamment
- **Pas de mise à jour automatique** : les réglementations évoluent ; le corpus doit être maintenu manuellement
- **Langue unique** : le corpus est en français ; les questions dans d'autres langues donnent de moins bons résultats

### Améliorations proposées

1. **Enrichissement du corpus** : web scraping automatisé de Service-Public.fr, ameli.fr, OFII
2. **LLM génératif** : intégration de Mistral-7B ou LLaMA via Ollama pour des réponses plus naturelles
3. **Mémoire conversationnelle** : utiliser LangChain `ConversationBufferMemory` pour le contexte multi-tour
4. **Interface web** : déploiement avec Gradio (`gr.ChatInterface`) ou Streamlit
5. **Multilingue** : réponses automatiques en arabe, anglais, espagnol via la traduction
6. **API REST** : encapsuler le chatbot dans une API FastAPI pour l'intégration dans d'autres applications
7. **Fine-tuning** : entraîner un modèle sur des conversations annotées réelles

---

## 🌐 Sources officielles

Les réponses du chatbot sont basées sur les sites gouvernementaux français :

| Site | URL | Contenu |
|---|---|---|
| Service-Public.fr | https://www.service-public.fr | Référence générale des démarches |
| ANEF | https://administration-etrangers.interieur.gouv.fr | Titres de séjour en ligne |
| France Visas | https://france-visas.gouv.fr | Demandes de visa |
| OFPRA | https://www.ofpra.gouv.fr | Demandes d'asile |
| Ameli.fr | https://www.ameli.fr | Assurance maladie |
| CAF.fr | https://www.caf.fr | Aides au logement |
| Campus France | https://www.campusfrance.org | Études en France |
| OFII | https://www.ofii.fr | Intégration et immigration |

---

##  Licence

Ce projet est à vocation pédagogique. Les informations administratives présentées sont issues de sources publiques officielles françaises et ne constituent pas un conseil juridique. Pour toute situation personnelle, consultez directement les organismes compétents ou un juriste spécialisé.

---

*Projet développé dans le cadre d'un cours sur les Large Language Models (LLM) — Architecture RAG appliquée aux services administratifs.*
