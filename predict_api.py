from fastapi import FastAPI, UploadFile
import numpy as np
import os
import shutil

# Essentia
from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
)

from labels import labels

app = FastAPI()


class Predictor:
    def __init__(self):
        # Fichiers pour la prédiction de genre
        self.embedding_model_file = "./models/discogs-effnet-bs64-1.pb"
        self.classification_model_file = "./models/genre_discogs400-discogs-effnet-1.pb"

        # Fichier pour la prédiction "approachability"
        self.approachability_model_file = (
            "./models/approachability_regression-discogs-effnet-1.pb"
        )

        self.sample_rate = 16000

        # Chargement Essentia
        self.loader = MonoLoader()

        # Modèle de genre (embeddings + classification)
        self.tensorflowPredictEffnetDiscogs = TensorflowPredictEffnetDiscogs(
            graphFilename=self.embedding_model_file,
            output="PartitionedCall:1",
            patchHopSize=128,
        )
        self.classification_model = TensorflowPredict2D(
            graphFilename=self.classification_model_file,
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )

        # Modèle "approachability"
        self.approachability_model = TensorflowPredict2D(
            graphFilename=self.approachability_model_file,
            output="model/Identity",
        )

    def check_model_files_exist(self):
        # Vérifie la présence de tous les fichiers
        return (
            os.path.exists(self.embedding_model_file)
            and os.path.exists(self.classification_model_file)
            and os.path.exists(self.approachability_model_file)
        )

    def load_audio(self, audio_path: str):
        """
        Charge l'audio en mémoire à partir du chemin spécifié.
        Retourne un tableau numpy contenant l'onde audio.
        """
        self.loader.configure(
            sampleRate=self.sample_rate,
            resampleQuality=4,
            filename=audio_path,
        )
        waveform = self.loader()
        return waveform

    def predict_genre(self, waveform):
        """
        Prédiction du genre principal, du genre complet et du genre secondaire
        à partir d'un waveform donné.
        """
        # Embeddings via EffNet
        embeddings = self.tensorflowPredictEffnetDiscogs(waveform)

        # Classification du genre
        activations = self.classification_model(embeddings)
        activations_mean = np.mean(activations, axis=0)

        # Traitement des résultats
        result_dict = dict(zip(labels, activations_mean.tolist()))
        sorted_genres = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)

        top_genre = sorted_genres[0][0]
        genre_primary, genre_full = map(str.strip, top_genre.split("---"))
        genre_secondary_full = sorted_genres[1][0]
        genre_secondary = genre_secondary_full.split("---")[1].strip()

        return genre_primary, genre_full, genre_secondary

    def predict_approachability(self, waveform):
        """
        Prédiction de la valeur d'« approachability » à partir d’un waveform.
        """
        # On réutilise les mêmes embeddings
        embeddings = self.tensorflowPredictEffnetDiscogs(waveform)
        # Modèle de régression "approachability"
        approachability_pred = self.approachability_model(embeddings)

        # Selon la forme de sortie, récupérez la valeur souhaitée.
        # Ici, on suppose une seule valeur en sortie (ex: shape = (1, 1)).
        # Adaptez en fonction de votre modèle réel.
        if approachability_pred.ndim == 2:
            return float(approachability_pred[0][0])
        else:
            return float(approachability_pred[0])

    def verify_models(self):
        """Vérifie la présence de tous les fichiers de modèles avant de prédire."""
        if not self.check_model_files_exist():
            raise FileNotFoundError(
                "Certains modèles n'existent pas. Assurez-vous de les avoir téléchargés."
            )


predictor = Predictor()


@app.post("/predict_genre/")
async def predict_genre(audio_file: UploadFile):
    """
    Endpoint pour prédire le genre principal, le genre complet et le genre secondaire.
    """
    # Vérification du type de fichier
    if not audio_file.filename.endswith((".mp3", ".wav")):
        return {"error": "Format de fichier non supporté. Envoyez un .mp3 ou un .wav."}

    # Sauvegarde temporaire
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_file.file.read())

    # Prédiction
    try:
        predictor.verify_models()
        waveform = predictor.load_audio(audio_path)
        genre_primary, genre_full, genre_secondary = predictor.predict_genre(waveform)
    finally:
        # Nettoyage
        os.remove(audio_path)

    return {
        "Primary Genre": genre_primary,
        "Full Genre": genre_full,
        "Secondary Genre": genre_secondary,
    }


@app.post("/predict_approachability/")
async def predict_approachability(audio_file: UploadFile):
    """
    Endpoint pour prédire la valeur « approachability ».
    """
    # Vérification du type de fichier
    if not audio_file.filename.endswith((".mp3", ".wav")):
        return {"error": "Format de fichier non supporté. Envoyez un .mp3 ou un .wav."}

    # Sauvegarde temporaire
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_file.file.read())

    # Prédiction
    try:
        predictor.verify_models()
        waveform = predictor.load_audio(audio_path)
        approachability_value = predictor.predict_approachability(waveform)
    finally:
        # Nettoyage
        os.remove(audio_path)

    return {
        "approachability_score": approachability_value
    }
