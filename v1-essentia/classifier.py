import os
import sqlite3
import hashlib
import datetime

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import essentia
import essentia.standard as es


# ------------------------------------------------------
# Database helpers
# ------------------------------------------------------
def create_tables_if_not_exist(conn: sqlite3.Connection):
    """
    Creates all necessary SQLite tables if they do not already exist.
    This replicates a schema similar to your original Drizzle-based setup.
    """
    cursor = conn.cursor()

    # audioFiles table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audioFiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filePath TEXT NOT NULL,
        fileName TEXT NOT NULL,
        fileHash TEXT NOT NULL UNIQUE,
        duration REAL,
        createdAt TEXT NOT NULL,
        updatedAt TEXT NOT NULL
    )
    """)

    # numericFeatures table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS numericFeatures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        audioFileId INTEGER NOT NULL,
        feature TEXT NOT NULL,
        value REAL NOT NULL
    )
    """)

    # categoricalFeatures table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categoricalFeatures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        audioFileId INTEGER NOT NULL,
        feature TEXT NOT NULL,
        value TEXT NOT NULL
    )
    """)

    # tags table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL
    )
    """)

    # audioFileTags table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audioFileTags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        audioFileId INTEGER NOT NULL,
        tagId INTEGER NOT NULL
    )
    """)

    conn.commit()

def insert_audio_file(
    conn: sqlite3.Connection,
    file_path: str,
    file_name: str,
    file_hash: str,
    duration: float
) -> int:
    """
    Inserts a new row in 'audioFiles' and returns the inserted row's ID.
    """
    now = datetime.datetime.utcnow().isoformat()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audioFiles (filePath, fileName, fileHash, duration, createdAt, updatedAt)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (file_path, file_name, file_hash, duration, now, now))
    conn.commit()
    return cursor.lastrowid

def insert_numeric_feature(
    conn: sqlite3.Connection,
    audio_file_id: int,
    feature: str,
    value: float
):
    """
    Inserts a numeric feature for a given audioFileId.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO numericFeatures (audioFileId, feature, value)
        VALUES (?, ?, ?)
    """, (audio_file_id, feature, value))
    conn.commit()

def insert_categorical_feature(
    conn: sqlite3.Connection,
    audio_file_id: int,
    feature: str,
    value: str
):
    """
    Inserts a categorical feature for a given audioFileId.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO categoricalFeatures (audioFileId, feature, value)
        VALUES (?, ?, ?)
    """, (audio_file_id, feature, value))
    conn.commit()

def get_or_create_tag(conn: sqlite3.Connection, name: str, category: str) -> int:
    """
    Retrieves an existing tag by (name, category). If not found, creates it.
    Returns the tag ID.
    """
    cursor = conn.cursor()
    # Check if the tag already exists
    cursor.execute("""
        SELECT id FROM tags
        WHERE name = ? AND category = ?
    """, (name, category))
    row = cursor.fetchone()

    if row:
        return row[0]  # existing tag ID

    # Otherwise, insert a new tag
    cursor.execute("""
        INSERT INTO tags (name, category)
        VALUES (?, ?)
    """, (name, category))
    conn.commit()
    return cursor.lastrowid

def insert_audio_file_tag(
    conn: sqlite3.Connection,
    audio_file_id: int,
    tag_id: int
):
    """
    Inserts a record in audioFileTags linking the audio file to the tag.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audioFileTags (audioFileId, tagId)
        VALUES (?, ?)
    """, (audio_file_id, tag_id))
    conn.commit()


# ------------------------------------------------------
# YAMNet label set (for demonstration; partial list)
# Full label set: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
# ------------------------------------------------------
YAMNET_CLASSES = [
    'Speech',
    'Male speech, man speaking',
    'Female speech, woman speaking',
    'Child speech, kid speaking',
    'Conversation',
    'Narration, monologue',
    'Music',
    'Guitar',
    'Piano',
    'Drum',
    'Violin, fiddle',
    'Bass guitar',
    # ...
]


# ------------------------------------------------------
# Python Audio Classifier
# ------------------------------------------------------
class AudioClassifier:
    def __init__(self, db_path: str, audio_dir: str):
        self.audio_dir = audio_dir
        self.db_path = db_path

        # 1) Initialize / connect to SQLite
        self.conn = sqlite3.connect(self.db_path)
        create_tables_if_not_exist(self.conn)

        # 2) Load YAMNet model from TF-Hub
        #    YAMNet is a TF2 SavedModel, so we can use hub.load
        print("Loading YAMNet model from TF-Hub...")
        self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("Model loaded.")

    def process_directory(self):
        """
        Walk the audio directory, process each audio file, store results to SQLite.
        """
        # Get all audio files in the directory
        files = [
            f for f in os.listdir(self.audio_dir)
            if f.lower().endswith(('.mp3', '.wav', '.m4a'))
        ]

        for file_name in files:
            file_path = os.path.join(self.audio_dir, file_name)

            # Compute file hash
            with open(file_path, 'rb') as f:
                file_content = f.read()
                file_hash = hashlib.sha256(file_content).hexdigest()

            # Check if this file was already processed by comparing hash
            if self.file_already_processed(file_hash):
                print(f"File {file_name} already processed; skipping.")
                continue

            # Perform classification
            print(f"Processing: {file_name}")
            classification = self.classify_audio(file_path)

            # Insert a new record in audioFiles
            audio_file_id = insert_audio_file(
                self.conn,
                file_path=file_path,
                file_name=file_name,
                file_hash=file_hash,
                duration=classification['numeric']['duration']
            )

            # Store numeric features
            for feat_name, feat_value in classification['numeric'].items():
                insert_numeric_feature(self.conn, audio_file_id, feat_name, feat_value)

            # Store categorical features
            for feat_name, feat_value in classification['categorical'].items():
                insert_categorical_feature(self.conn, audio_file_id, feat_name, feat_value)

            # Store tags
            for category, tag_list in classification['tags'].items():
                for tag_name in tag_list:
                    tag_id = get_or_create_tag(self.conn, tag_name, category)
                    insert_audio_file_tag(self.conn, audio_file_id, tag_id)

    def file_already_processed(self, file_hash: str) -> bool:
        """
        Checks if there's an entry in audioFiles with the given hash.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id FROM audioFiles WHERE fileHash = ?
        """, (file_hash,))
        row = cursor.fetchone()
        return row is not None

    def classify_audio(self, file_path: str):
        """
        Load & analyze the audio using Essentia, run YAMNet inference, then build a
        classification dict with numeric, categorical, and tag data.
        """
        # 1) Load audio at 44.1k or whatever sample rate you want for feature extraction
        loader = es.MonoLoader(filename=file_path, sampleRate=44100)
        audio_data = loader()

        # 2) Extract features with Essentia
        features = self.extract_essentia_features(audio_data, sr=44100)

        # 3) Run YAMNet on 16k resampled audio
        yamnet_labels = self.run_yamnet_inference(audio_data, orig_sr=44100, target_sr=16000)

        # 4) Build classification object
        classification = {
            'numeric': {
                'energy': features['energy'],
                'tempo': features['bpm'],
                'loudness': features['loudness'],
                'duration': len(audio_data) / 44100.0
            },
            'categorical': {
                'genre': self.determine_genre(yamnet_labels, features),
                'key': features['key'],
                'time_signature': '4/4'  # Example: if you want real detection, see RhythmExtractor2013
            },
            'tags': self.map_yamnet_labels_to_tags(yamnet_labels, features)
        }

        return classification

    def extract_essentia_features(self, audio_data: np.ndarray, sr: int = 44100):
        """
        Extract some example features using Essentia's standard algorithms.
        """
        # Energy
        energy_algo = es.Energy()
        total_energy = energy_algo(audio_data)

        # BPM (RhythmExtractor2013 returns multiple values)
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, confidence, estimates, bpm_intervals, beats = rhythm_extractor(audio_data)

        # Loudness (LUFS-based)
        loudness_extractor = es.Loudness()
        loudness_val = loudness_extractor(audio_data)

        # Key detection
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio_data)

        features = {
            'energy': float(total_energy),
            'bpm': float(bpm),
            'loudness': float(loudness_val),
            'key': f"{key} {scale}"
        }
        return features

    def run_yamnet_inference(self, audio_data: np.ndarray, orig_sr=44100, target_sr=16000, top_k=5):
        """
        Resample audio to 16k, run YAMNet. Return top-K label strings.
        """
        # 1) Resample from orig_sr to target_sr (16k)
        if orig_sr != target_sr:
            resample = es.Resample(inputSampleRate=orig_sr, outputSampleRate=target_sr)
            audio_16k = resample(audio_data)
        else:
            audio_16k = audio_data

        # 2) Convert to float32 Tensor
        waveform = tf.constant(audio_16k, dtype=tf.float32)

        # 3) Run the model. YAMNet returns scores, embeddings, spectrogram
        #    - scores shape: [N_FRAMES, 521] for AudioSet classification
        #    - embeddings shape: [N_FRAMES, 1024]
        #    - log_mel_spectrogram shape: [N_FRAMES, 64]
        scores, embeddings, spectrogram = self.yamnet_model(waveform)

        # 4) Average scores across frames -> shape [521]
        mean_scores = tf.reduce_mean(scores, axis=0)

        # 5) Get top_k class indices
        top_indices = tf.argsort(mean_scores, direction='DESCENDING')[:top_k].numpy()
        top_labels = [YAMNET_CLASSES[i] if i < len(YAMNET_CLASSES) else f"Class_{i}"
                      for i in top_indices]

        return top_labels

    def map_yamnet_labels_to_tags(self, labels, features):
        """
        Map YAMNet labels + extracted features to your custom tag categories.
        Returns a dict like: { 'instrument': [], 'mood': [], 'vocals': [] }
        """
        tags = {
            'instrument': [],
            'mood': [],
            'vocals': []
        }

        INSTRUMENT_KEYWORDS = ['guitar', 'drum', 'piano', 'bass', 'violin']
        VOCAL_KEYWORDS = ['singing', 'male speech', 'female speech', 'speech']

        for label in labels:
            low_label = label.lower()
            if any(k in low_label for k in INSTRUMENT_KEYWORDS):
                tags['instrument'].append(label)
            if any(k in low_label for k in VOCAL_KEYWORDS):
                tags['vocals'].append(label)

        # Example mood tag
        if features['energy'] > 0.8:
            tags['mood'].append('energetic')
        elif features['energy'] < 0.3:
            tags['mood'].append('calm')

        return tags

    def determine_genre(self, yamnet_labels, features):
        """
        Determine a 'genre' from YAMNet labels and features.
        Very naive, purely an example.
        """
        genre_keywords = {
            'rock': ['electric guitar', 'drum kit', 'distortion'],
            'classical': ['violin', 'orchestra'],
            'electronic': ['synth', 'electronic'],
            'jazz': ['saxophone', 'jazz guitar', 'trumpet']
        }

        # Count matches
        label_set = set(l.lower() for l in yamnet_labels)
        best_genre = 'unknown'
        best_score = 0

        for genre, keywords in genre_keywords.items():
            score = sum(k in ' '.join(label_set) for k in keywords)
            if score > best_score:
                best_score = score
                best_genre = genre

        return best_genre


# ------------------------------------------------------
# Usage Example
# ------------------------------------------------------
if __name__ == "__main__":
    # Create classifier instance
    #  - db_path: where SQLite DB is stored
    #  - audio_dir: directory with .mp3/.wav/.m4a files
    classifier = AudioClassifier(db_path="audio_classification.db", audio_dir="./audio")

    # Process all audio files in the given directory
    classifier.process_directory()
    print("Done processing audio directory.")