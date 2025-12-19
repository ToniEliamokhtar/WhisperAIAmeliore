# WhisperAIAmélioré

Projet EXTRA fait par moi, Toni Eliamokhtar, pour monsieur Thomas Piquet

Dans ce projet, je vais utiliser Open Web UI (vous m'en avez parlé lors de ma démo) pour implémenter deux outils en lien avec Whisper AI pour générer un spectogramme et analyser un fichier audio

> Repo initialisé proprement avec Docker et Git avant le développement, je vais utiliser Docker compose pour pouvoir lancer deux containers en même temps (celui de open web ui, et celui de whisper)

> le gitignore je l'ai fait après une petite recherche (youtube, sites web, mais je ne vais pas tout expliquer les sources et comment j'ai fait car c'est un travail extra que je fais juste pour vous montrer que je peux vraiment faire quelque chose de plus pour vous avec whisper), mais on verra si ça va vraiment aider (les fichiers incluls)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Prérequis :
1. Docker compose (si vous avez Docker desktop ça devrait être correct, sinon , vérifier si vous l'avez et l'installer au besoin)
2. Ajouter llama3 de Ollama comme modèle dans OpenWebUi
3. Ajouter les outils Whisper Audio Analyzer et Whisper Spectogram dans OpenWebUi --> Workspace/Espace de travail --> Outils --> Ajouter (Pour le code, je vais le donner à la fin du Readme)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Installation :
git clone https://github.com/ToniEliamokhtar/WhisperAIAmeliore.git
cd WhisperAIAmeliore
docker compose up -d --build

- Le premier lancement peut être plus long (téléchargement de Whisper et du modèle LLaMA via Ollama)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Vérification du service (voir si ça marche) :
curl http://localhost:8000/health
    Résultat attendu : {"status":"ok","model":"base"}

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Utilisation avec OpenWebUI :
1. Ouvrir : http://localhost:3000
2. Choisir un modèle (ex. llama3)
3. Utiliser les outils personnalisés (voir en bas pour le code des outils)
4. Lancer le prompt :


Analyse le fichier audio suivant : /samples/normalEnglish.wav

1) Utilise l’outil **Whisper Audio Analyzer** pour :
- détecter la langue
- transcrire le texte
- fournir la traduction anglaise

2) Utilise ensuite l’outil **Whisper Spectrogramme** pour :
- générer le spectrogramme audio
- afficher l’image directement dans la conversation

Présente les résultats de façon claire.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Whisper Audio Analyzer :

import os
import requests
from pydantic import Field


class Tools:
    def __init__(self):
        pass

    def analyze_audio(
        self,
        file_path: str = Field(
            ...,
            description="Path to the audio file inside OpenWebUI (ex: /samples/normalEnglish.wav).",
        ),
    ) -> str:
        """
        Transcribe audio, detect language and translate to English using Whisper.
        """

        # 1) Validate path
        if not file_path or not isinstance(file_path, str):
            return "Error: file_path is empty."

        if not os.path.isfile(file_path):
            return (
                "Error: file not found.\n"
                f"Path given: {file_path}\n"
                "Tip: use a path like /samples/normalEnglish.wav"
            )

        # 2) Call service
        url = "http://whisper-audio-service:8000/analyze"

        try:
            with open(file_path, "rb") as audio_file:
                files = {"file": (os.path.basename(file_path), audio_file)}
                response = requests.post(url, files=files, timeout=120)
                response.raise_for_status()
                data = response.json()

            language = data.get("language", "unknown")
            original = data.get("text_original", "")
            english = data.get("text_english", "")

            return (
                f"Language: {language}\n\n"
                f"Original text:\n{original}\n\n"
                f"English translation:\n{english}"
            )

        except requests.exceptions.Timeout:
            return "Error: Whisper service timed out (took too long)."
        except requests.exceptions.RequestException as e:
            return f"Error: request to Whisper service failed: {str(e)}"
        except Exception as e:
            return f"Error analyzing audio: {str(e)}"



Whisper Spectogram :

import os
import requests
from pydantic import Field


class Tools:
    def __init__(self):
        pass

    def generate_spectrogram(
        self,
        file_path: str = Field(
            ...,
            description="Chemin du fichier audio accessible dans le container OpenWebUI (ex: /samples/normalEnglish.wav).",
        ),
    ) -> str:
        """
        Génère un spectrogramme via le Whisper Audio Service et retourne un aperçu (image) + un lien.
        """
        api_base = os.getenv("PUBLIC_AUDIO_SERVICE_URL", "http://localhost:8000")
        api_internal = "http://whisper-audio-service:8000"

        url = f"{api_internal}/spectrogram"

        try:
            with open(file_path, "rb") as audio_file:
                files = {"file": (os.path.basename(file_path), audio_file)}
                r = requests.post(url, files=files, timeout=120)
                r.raise_for_status()
                data = r.json()

            image_url = data.get("image_url")
            if not image_url:
                return f"Erreur: aucune URL d'image reçue.\nRéponse: {data}"

            # Si ton API retourne localhost, c'est correct pour ton navigateur (tu es en local).
            # Mais au cas où, on remplace la base si besoin.
            if image_url.startswith("http://localhost:8000"):
                image_url = image_url.replace("http://localhost:8000", api_base)

            return (
                f"✅ Spectrogramme généré.\n\n"
                f"**Aperçu :**\n\n"
                f"![Spectrogramme]({image_url})\n\n"
                f"**Lien direct :** {image_url}"
            )

        except FileNotFoundError:
            return f"Erreur: fichier introuvable: {file_path}\nAssure-toi que /samples est bien monté."
        except requests.RequestException as e:
            return f"Erreur HTTP: {str(e)}"
        except Exception as e:
            return f"Erreur: {str(e)}"

