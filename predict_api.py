from fastapi import FastAPI, UploadFile, Query
import numpy as np

from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
)
import os
import shutil

from labels import labels

app = FastAPI()


class Predictor:
    def __init__(self):
        self.embedding_model_file = "./models/discogs-effnet-bs64-1.pb"
        self.classification_model_file = "./models/genre_discogs400-discogs-effnet-1.pb"
        self.approachability_2c_model_file = "./models/approachability_2c-discogs-effnet-1.pb"
        self.approachability_3c_model_file = "./models/approachability_3c-discogs-effnet-1.pb"
        self.approachability_regression_model_file = "./models/approachability_regression-discogs-effnet-1.pb"

        self.output = "activations"
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
        self.approachability_2c_model = TensorflowPredict2D(
            graphFilename=self.approachability_2c_model_file,
            output="model/Identity",
        )
        self.approachability_3c_model = TensorflowPredict2D(
            graphFilename=self.approachability_3c_model_file,
            output="model/Identity",
        )
        self.approachability_regression_model = TensorflowPredict2D(
            graphFilename=self.approachability_regression_model_file,
            output="model/Identity",
        )

    def check_model_files_exist(self):
        return all(
            os.path.exists(model_file)
            for model_file in [
                self.embedding_model_file,
                self.classification_model_file,
                self.approachability_2c_model_file,
                self.approachability_3c_model_file,
                self.approachability_regression_model_file,
            ]
        )

    def load_audio_from_file(self, file):
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as audio_file:
            shutil.copyfileobj(file, audio_file)
        return audio_path

    def predict(self, audio_path, approachability_type="regression"):
        if not self.check_model_files_exist():
            raise FileNotFoundError(
                "Model files do not exist. Please download them using download.sh script."
            )

        print("Loading audio...")
        self.loader.configure(
            sampleRate=self.sample_rate,
            resampleQuality=4,
            filename=audio_path,
        )
        waveform = self.loader()

        # Model Inferencing
        print("Running the model...")
        embeddings = self.tensorflowPredictEffnetDiscogs(waveform)
        activations = self.classification_model(embeddings)
        activations_mean = np.mean(activations, axis=0)

        # Parsing Genres
        result_dict = dict(zip(labels, activations_mean.tolist()))
        sorted_genres = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        top_genre = sorted_genres[0][0]
        genre_primary, genre_full = map(str.strip, top_genre.split("---"))
        genre_secondary_full = sorted_genres[1][0]
        genre_secondary = genre_secondary_full.split("---")[1].strip()

        # Predicting Approachability
        if approachability_type == "2c":
            approachability = self.approachability_2c_model(embeddings).tolist()
        elif approachability_type == "3c":
            approachability = self.approachability_3c_model(embeddings).tolist()
        elif approachability_type == "regression":
            approachability = self.approachability_regression_model(embeddings).tolist()
        else:
            raise ValueError("Invalid approachability type. Choose from '2c', '3c', or 'regression'.")

        return {
            "Primary Genre": genre_primary,
            "Full Genre": genre_full,
            "Secondary Genre": genre_secondary,
            "Approachability": approachability,
        }


predictor = Predictor()


@app.post("/predict/")
async def predict_genre_and_approachability(
    audio_file: UploadFile,
    approachability_type: str = Query("regression", enum=["2c", "3c", "regression"]),
):
    # Check if the uploaded file is an audio file
    if not audio_file.filename.endswith((".mp3", ".wav")):
        return {"error": "Invalid file type. Please upload a .mp3 or .wav file."}

    # Save the uploaded file temporarily
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as audio_data:
        audio_data.write(audio_file.file.read())

    try:
        result = predictor.predict(audio_path, approachability_type)
    finally:
        # Clean up temporary audio file
        os.remove(audio_path)

    return result
