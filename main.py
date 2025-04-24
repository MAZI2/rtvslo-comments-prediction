import yaml
import json
import argparse
import os
import random
from itertools import cycle

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap.umap_ import UMAP

import plotly.graph_objects as go
import plotly.colors as pc

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import yake

# Set all possible random sources to have deterministic results (important for caching)
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

###################
# Data processing #
###################

# Load YAML data
def load_articles(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        articles = yaml.safe_load(file)
    return articles

# Extract representative texts from articles
def get_article_text(article):
    text_parts = [
        article.get('title', ''),
        article.get('lead', ''),
        ' '.join(article.get('paragraphs', [])),
    ]
    return ' '.join(text_parts)

def load_preprocessed(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

##############
# Clustering #
##############

def cluster_with_kmeans(embeddings, n_clusters=20, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    clusters = kmeans.fit_predict(embeddings)
    return clusters


######################
# Keyword extraction #
######################

# YAKE Keyword Extraction
def extract_yake_keywords(text, top_n=20, language='sl'):
    kw_extractor = yake.KeywordExtractor(lan=language, top=top_n)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, _ in keywords]

# KeyBERT Keyword Extraction
def extract_keybert_keywords(keybert_model, text, stop_words, top_n=20):
    keywords = keybert_model.extract_keywords(text, top_n=top_n, stop_words=stop_words)
    return [kw for kw, _ in keywords]

# TF-IDF Keyword Extraction
def extract_tfidf_keywords(cluster_texts, stop_words, top_n=20, ngram_range=(1, 2)):
    # TF-IDF with stopword filtering and n-grams
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=stop_words,
        ngram_range=ngram_range,
        token_pattern=r'\b\w{3,}\b',  # only tokens with 3+ letters
    )
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

    # Top N terms by TF-IDF score
    top_indices = tfidf_scores.argsort()[::-1][:top_n]
    top_keywords = feature_array[top_indices]
    return list(top_keywords)

def extract_cluster_keywords(texts_tokenized, clusters, stop_words, keybert_model):
    """
    Extracts top n-grams (unigrams + bigrams) from preprocessed texts by cluster.
    """
    unique_clusters = np.unique(clusters[clusters >= 0])
    keywords_per_cluster = {}

    for cluster_label in unique_clusters:
        cluster_texts = [preprocessed_texts[i] for i in range(len(texts_tokenized)) if clusters[i] == cluster_label]

        tfidf_keywords = extract_tfidf_keywords(cluster_texts, stop_words)
        yake_keywords = extract_yake_keywords(" ".join(cluster_texts))
        keybert_keywords = extract_keybert_keywords(keybert_model, " ".join(cluster_texts), stop_words)

        print("cluster ", cluster_label, " done")
        # print("tfidf: ", tfidf_keywords)
        # print("yake: ", yake_keywords)
        # print("keybert: ", keybert_keywords)

        keywords_per_cluster[cluster_label] = {
            "tfidf": tfidf_keywords,
            "keybert": keybert_keywords,
            "yake": yake_keywords
        }

    return keywords_per_cluster

#################
# Visualization #
#################

def draw_plot(
    umap_embeddings,
    clusters,
    titles=None,
    keywords_per_cluster=None,
    show_group_outlines=True,
    title="Vizualizacija novic rtvslo.si"
):
    import re

    def hex_to_rgb(color):
        if color.startswith("rgb"):
            return tuple(map(int, re.findall(r'\d+', color)))
        color = color.lstrip('#')
        lv = len(color)
        return tuple(int(color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def get_filtered_points(points, percentile=95):
        center = np.median(points, axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        threshold = np.percentile(dists, percentile)
        return points[dists <= threshold]

    def fit_ellipse_to_points(points, scale=2.4477):
        pca = PCA(n_components=2)
        pca.fit(points)
        center = pca.mean_
        width, height = np.sqrt(pca.explained_variance_) * scale
        angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))

        t = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = center[0] + width * np.cos(t) * np.cos(np.radians(angle)) - height * np.sin(t) * np.sin(np.radians(angle))
        ellipse_y = center[1] + width * np.cos(t) * np.sin(np.radians(angle)) + height * np.sin(t) * np.cos(np.radians(angle))
        return ellipse_x, ellipse_y

    def summarize_keywords(keywords_dict, short=True, max_words=3, size=None, cluster_id=None):
        tfidf = keywords_dict.get("tfidf", [])
        tfidf_preview = ", ".join(tfidf[:max_words])
        if short:
            return tfidf_preview
        else:
            lines = [
                f"Skupina <b>{cluster_id}</b>",
                f"velikost: <b>{size}</b> novic",
                "―" * 20,
                f"<b>tfidf</b>: {', '.join(tfidf[:10])}",
                "―" * 20,
                f"<b>keybert</b>: {', '.join(keywords_dict.get('keybert', [])[:10])}",
                f"<b>yake</b>: {', '.join(keywords_dict.get('yake', [])[:5])}",
            ]
            return "<br>".join(lines)

    # Create main dataframe
    df = pd.DataFrame({
        'x': umap_embeddings[:, 0],
        'y': umap_embeddings[:, 1],
        'cluster': clusters.astype(str),
        'title': titles,
    })

    # Generate short and full label maps
    cluster_sizes = df['cluster'].value_counts().to_dict()
    
    if keywords_per_cluster:
        short_names = {
            cluster_id: summarize_keywords(
                keywords,
                short=True,
                size=cluster_sizes.get(str(cluster_id), 0),
                cluster_id=cluster_id
            )
            for cluster_id, keywords in keywords_per_cluster.items()
        }
        hover_texts = {
            cluster_id: summarize_keywords(
                keywords,
                short=False,
                size=cluster_sizes.get(str(cluster_id), 0),
                cluster_id=cluster_id
            )
            for cluster_id, keywords in keywords_per_cluster.items()
        }
    else:
        short_names = {}
        hover_texts = {}

    
    # Assign group label
    if keywords_per_cluster:
        df['group'] = [f"Skupina {int(c)}: {short_names.get(int(c), 'Unknown')}" for c in clusters]
    else:
        df['group'] = df['cluster']

    unique_groups = sorted(df['group'].unique(), key=lambda g: int(g.split(":")[0].replace("Skupina", "").strip()))

    group_to_cluster = {
        g: int(g.split(":")[0].replace("Skupina", "").strip())
        for g in unique_groups
    }

    color_pool = (
        pc.qualitative.Plotly +
        pc.qualitative.D3 +
        pc.qualitative.Set1 +
        pc.qualitative.Set2 +
        pc.qualitative.Set3 +
        pc.qualitative.Pastel1 +
        pc.qualitative.Pastel2 +
        pc.qualitative.Dark24 +
        pc.qualitative.Alphabet
    )
    color_cycle = cycle(color_pool)
    group_to_color = {group: next(color_cycle) for group in unique_groups}

    fig = go.Figure()

    for group in unique_groups:
        sub_df = df[df['group'] == group]
        fig.add_trace(
            go.Scattergl(
                x=sub_df['x'],
                y=sub_df['y'],
                mode='markers',
                name=group,
                marker=dict(size=4, color=group_to_color[group]),
                legendgroup=group,
                showlegend=True,
                hoverinfo='skip',
            )
        )

        if show_group_outlines and len(sub_df) >= 5:
            points = sub_df[['x', 'y']].to_numpy(dtype=np.float32)
            filtered_points = get_filtered_points(points, percentile=80)

            if len(filtered_points) >= 5:
                ellipse_x, ellipse_y = fit_ellipse_to_points(filtered_points)

                rgb = hex_to_rgb(group_to_color[group])
                rgba_fill = f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.1)'

                fig.add_trace(
                    go.Scattergl(
                        x=ellipse_x,
                        y=ellipse_y,
                        mode='lines',
                        line=dict(color=group_to_color[group], width=1),
                        fill='toself',
                        fillcolor=rgba_fill,
                        name=group,
                        legendgroup=group,
                        hoverinfo='text',
                        text=[hover_texts.get(group_to_cluster[group], group)] * len(ellipse_x),
                        showlegend=False
                    )
                )

                # Add cluster ID label at center
                cluster_id = group_to_cluster[group]
                fig.add_annotation(
                    x=np.mean(filtered_points[:, 0]),
                    y=np.mean(filtered_points[:, 1]),
                    text=f"Skupina {cluster_id}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="white",
                    opacity=0.9
                )

    fig.update_layout(
        title=title,
        width=1200,
        height=800,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.40,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=10)
        ),
        margin=dict(t=80, b=80)
    )

    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    group.add_argument("--no_cache", action="store_true", help="Disable all caching")

    args = parser.parse_args()

    # Defaults
    cached = True

    if args.no_cache:
        cached = False

    if not cached:
        print("downloading necessary stuff...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        stop_words = set(stopwords.words('slovene'))

        print("loading articles...")
        articles = load_articles('articles.yaml')
        texts = [get_article_text(article) for article in tqdm(articles, desc="Processing articles")]

        sbert_embeddings = np.load('cached/sbert_embeddings.npy')

        print("UMAP on SBERT embeddings...")
        umap_25d = UMAP(n_components=25, metric='cosine', random_state=42).fit_transform(sbert_embeddings)
        umap_2d = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(sbert_embeddings)

        #np.save("umap_2d.npy", umap_2d)

        sbert_clusters = cluster_with_kmeans(umap_25d, n_clusters=15, random_state=46)

        preprocessed_texts = load_preprocessed("cached/preprocessed_combined.jsonl")
        tokenized_texts = [text.split() for text in preprocessed_texts]
    else:
        keywords_per_cluster = np.load("cached/keywords_per_cluster.npy", allow_pickle=True).item()
        sbert_clusters = np.load("cached/sbert_clusters.npy")
        umap_2d = np.load("cached/umap_2d.npy")

    if not cached:
        print("extracting keywords...")
        print("downloading keybert...")
        keybert_model = KeyBERT(model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        keywords_per_cluster = extract_cluster_keywords(tokenized_texts, sbert_clusters, list(stop_words), keybert_model)

        #np.save("sbert_clusters_final.npy", sbert_clusters)
        #np.save("keywords_final.npy", keywords_per_cluster)

    draw_plot(
    umap_embeddings=umap_2d,
    clusters=sbert_clusters,
    keywords_per_cluster=keywords_per_cluster,
    show_group_outlines=True
    )