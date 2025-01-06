from fastapi import FastAPI, UploadFile
import numpy as np
import os
import shutil

from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
)
from labels import labels

app = FastAPI()


class Predictor:
    def __init__(self):
        self.embedding_model_file = "./models/discogs-effnet-bs64-1.pb"
        self.classification_model_file = "./models/genre_discogs400-discogs-effnet-1.pb"
        self.approachability_model_file = "./models/approachability_regression-discogs-effnet-1.pb"
        self.sample_rate = 16000

        self.loader = MonoLoader()
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
        self.approachability_model = TensorflowPredict2D(
            graphFilename=self.approachability_model_file,
            output="model/Identity",
        )

    def check_model_files_exist(self):
        required_files = [
            self.embedding_model_file,
            self.classification_model_file,
            self.approachability_model_file,
        ]
        return all(os.path.exists(file) for file in required_files)

    def load_audio_from_file(self, file):
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as audio_file:
            shutil.copyfileobj(file, audio_file)
        return audio_path

    def predict(self, audio_path):
        if not self.check_model_files_exist():
            raise FileNotFoundError(
                "Model files do not exist. Please ensure all model files are present."
            )

        print("Loading audio...")
        self.loader.configure(
            sampleRate=self.sample_rate,
            resampleQuality=4,
            filename=audio_path,
        )
        waveform = self.loader()

        # Embedding extraction
        print("Generating embeddings...")
        embeddings = self.tensorflowPredictEffnetDiscogs(waveform)

        # Genre classification
        print("Classifying genres...")
        activations = self.classification_model(embeddings)
        activations_mean = np.mean(activations, axis=0)

        # Parsing Genres
        result_dict = dict(zip(labels, activations_mean.tolist()))
        sorted_genres = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        top_genre = sorted_genres[0][0]
        genre_primary, genre_full = map(str.strip, top_genre.split("---"))
        genre_secondary_full = sorted_genres[1][0]
        genre_secondary = genre_secondary_full.split("---")[1].strip()

        return genre_primary, genre_full, genre_secondary, embeddings

    def predict_approachability(self, embeddings):
        print("Predicting approachability...")
        predictions = self.approachability_model(embeddings)
        return predictions.tolist()


predictor = Predictor()


@app.post("/predict/")
async def predict_genre(audio_file: UploadFile):
    if not audio_file.filename.endswith((".mp3", ".wav")):
        return {"error": "File format not supported. Please upload a .mp3 or .wav file."}

    # Save the uploaded file temporarily
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as audio_data:
        audio_data.write(audio_file.file.read())

    try:
        genre_primary, genre_full, genre_secondary, embeddings = predictor.predict(audio_path)
        approachability_score = predictor.predict_approachability(embeddings)
    finally:
        # Clean up temporary audio file
        os.remove(audio_path)

    return {
        "Primary Genre": genre_primary,
        "Full Genre": genre_full,
        "Secondary Genre": genre_secondary,
        "Approachability Score": approachability_score,
    }
