import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from utils.db import select
import re

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

K = 3                 # neighbors used for k-distance plot / min_samples default
EPS = 0.5             # <-- set this after inspecting the k-distance plot's elbow
MIN_SAMPLES = K       # override independently if desired
MODEL_NAME = "all-MiniLM-L6-v2"

def embed_sentences(sentences, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return np.asarray(embeddings)


def plot_k_distance(embeddings, k=K):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    k_distances = np.sort(distances[:, -1])  # distance to the k-th neighbor

    plt.figure()
    plt.plot(k_distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}-th nearest neighbor")
    plt.title("k-distance graph (look for the elbow to pick eps)")
    plt.show()


def run_dbscan(embeddings, eps=EPS, min_samples=MIN_SAMPLES):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(embeddings)
    return labels


def plot_clusters_2d(embeddings, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure()
    noise_mask = labels == -1
    plt.scatter(
        reduced[~noise_mask, 0], reduced[~noise_mask, 1],
        c=labels[~noise_mask], cmap="tab10", marker="o", label="clustered"
    )
    plt.scatter(
        reduced[noise_mask, 0], reduced[noise_mask, 1],
        c="black", marker="x", label="noise/outlier"
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("DBSCAN clusters (PCA projection, for viz only)")
    plt.legend()
    plt.show()

def read_file(filename):
    try:
        with open(filename, "r") as file:
            text = file.read()
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return sentences
    except Exception as e:
        print(f"Failed to Read file {filename}, {e}")
    return ""

def run(gen_id, engine):

    SessionLocal = select.initiate_sessionlocal(engine)
    iteration_details = select.fetch_iteration_id_using_gen_id(
        SessionLocal,
        gen_id
    ).all()

    iteration_id = []
    for iteration in iteration_details:
        iteration_id.append(iteration.iteration_id)

    iteration_id.sort()
    file_path = []
    embeddings = []

    for i in range(len(iteration_id)):
        file_path.append(select.fetch_documents_per_iteration(
            SessionLocal,
            iteration_id[i]
        ).all()[0].file_path)

    for file in file_path:
        sentences = read_file(file)
        if sentences:
            for sen in sentences:
                embedding = embed_sentences(sen)
                embeddings.append(embedding)

    plot_k_distance(embeddings, k=K)

    labels = run_dbscan(embeddings, eps=EPS, min_samples=MIN_SAMPLES)

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    print(f"Total sentences: {len(sentences)}")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise/outlier points: {n_noise}")
    print()
    print("Outlier sentences:")
    for sentence, label in zip(sentences, labels):
        if label == -1:
            print(f"  - {sentence.strip()}")

    plot_clusters_2d(embeddings, labels)
