import os
import subprocess
import gdown
import librosa
import audioread
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as preprocess_inception,
)
from tqdm import tqdm


def download_video(url, path):
    try:
        file_id_video = url.split("/")[5]
        download_url_video = f"https://drive.google.com/uc?id={file_id_video}"
        gdown.download(download_url_video, path, quiet=False)
        print(f"Video downloaded to: {path}")
    except Exception as e:
        print(f"Error downloading video: {e}")


def check_video_audio(video_path):
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if "Audio:" in result.stdout:
            print(f"Audio stream found in {video_path}")
            return True
        else:
            print(f"No audio stream found in {video_path}")
            return False
    except Exception as e:
        print(f"Error checking audio stream in {video_path}: {e}")
        return False


def load_audio(video_path):
    try:
        if not os.path.exists(video_path):
            print(f"Video path does not exist: {video_path}")
            return None, None
        with audioread.audio_open(video_path) as input_file:
            sr = input_file.samplerate
            y = np.concatenate(
                [np.frombuffer(buf, dtype=np.int16) for buf in input_file]
            )
            y = librosa.util.buf_to_float(y, n_bytes=2, dtype=np.float32)
        return y, sr
    except Exception as e:
        print(f"Error loading audio from {video_path}: {e}")
        return None, None


def extract_audio_features(video_path):
    y, sr = load_audio(video_path)
    if y is None or sr is None:
        return None
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    return np.concatenate(
        (mfccs, chroma, spectral_contrast, tonnetz, mel_spectrogram, zero_crossings),
        axis=0,
    )


def calculate_audio_similarity(features1, features2):
    min_length = min(features1.shape[1], features2.shape[1])
    features1 = features1[:, :min_length]
    features2 = features2[:, :min_length]
    distances = [cosine(f1, f2) for f1, f2 in zip(features1.T, features2.T)]
    similarity = 1 - np.mean(distances)
    return similarity * 100


def extract_visual_features(video_path, shape, model, batch_size=32):
    cap = cv2.VideoCapture(video_path)
    features = []
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count), desc="Processing Video Frames"):
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, shape)
            img = image.img_to_array(img)
            if model.name == "resnet50":
                img = preprocess_resnet(img)
            elif model.name == "inception_v3":
                img = preprocess_inception(img)
            frames.append(img)
            if len(frames) == batch_size:
                batch = np.array(frames)
                batch_features = model.predict(batch)
                features.extend(batch_features)
                frames = []
        else:
            break

    if frames:
        batch = np.array(frames)
        batch_features = model.predict(batch)
        features.extend(batch_features)

    cap.release()
    return np.array(features)


def calculate_visual_similarity(features1, features2):
    min_length = min(len(features1), len(features2))
    features1 = features1[:min_length]
    features2 = features2[:min_length]
    distances = [cosine(f1, f2) for f1, f2 in zip(features1, features2)]
    similarity = 1 - np.mean(distances)
    return similarity * 100


def calculate_overall_similarity(audio_similarity, visual_similarity):
    return (audio_similarity + visual_similarity) / 2


def run_video_similarity_model(
    model_name, model, shape, audio_similarity, video1_path, video2_path
):
    try:
        visual_features1 = extract_visual_features(video1_path, shape, model)
        visual_features2 = extract_visual_features(video2_path, shape, model)

        visual_similarity = calculate_visual_similarity(
            visual_features1, visual_features2
        )
        overall_similarity_percentage = calculate_overall_similarity(
            audio_similarity, visual_similarity
        )

        print(f"Model : {model_name}")
        print(f"Visual Similarity: {visual_similarity:.2f}%")
        print(f"Overall Similarity: {overall_similarity_percentage:.2f}%")

        return model_name, visual_similarity, overall_similarity_percentage
    except Exception as e:
        print(f"Error running video similarity model: {e}")
        return model_name, 0, 0


def main():
    sharing_link_video1 = "https://drive.google.com/file/d/1vrz_EJ3D1j6yYi5NnlTYssfGoyq1vqnw/view?usp=drive_link"
    sharing_link_video2 = "https://drive.google.com/file/d/1sb5ODQcGu-N3f_6XllVTBAsMzRaMGWEn/view?usp=drive_link"
    path = "./videos/"

    if not os.path.exists(path):
        os.makedirs(path)

    video1_path = os.path.join(path, "video1.mp4")
    video2_path = os.path.join(path, "video2.mp4")

    print("Downloading videos...")
    download_video(sharing_link_video1, video1_path)
    download_video(sharing_link_video2, video2_path)

    if not check_video_audio(video1_path) or not check_video_audio(video2_path):
        print("One or both videos do not contain audio streams. Exiting...")
        return

    print("Extracting audio features...")
    audio_features1 = extract_audio_features(video1_path)
    audio_features2 = extract_audio_features(video2_path)

    if audio_features1 is None or audio_features2 is None:
        print("Error extracting audio features. Exiting...")
        return

    audio_similarity = calculate_audio_similarity(audio_features1, audio_features2)

    results = []

    print("Running ResNet50 model...")
    input_shape = (224, 224, 3)
    shape = (224, 224)
    resnet_model = ResNet50(
        weights="imagenet", include_top=False, pooling="avg", input_shape=input_shape
    )

    result = run_video_similarity_model(
        "ResNet50", resnet_model, shape, audio_similarity, video1_path, video2_path
    )
    results.append(result)

    print("Running InceptionV3 model...")
    input_shape = (299, 299, 3)
    shape = (299, 299)
    inception_model = InceptionV3(
        weights="imagenet", include_top=False, pooling="avg", input_shape=input_shape
    )

    result = run_video_similarity_model(
        "InceptionV3",
        inception_model,
        shape,
        audio_similarity,
        video1_path,
        video2_path,
    )
    results.append(result)

    df = pd.DataFrame(
        results, columns=["Model", "Visual Similarity", "Overall Similarity"]
    )

    print(f"Audio Similarity: {audio_similarity:.2f}%")

    print("\nModel Performance Table for Video Similarity:")
    print(df)


if __name__ == "__main__":
    main()
