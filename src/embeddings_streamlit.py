import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
from sentence_transformers import SentenceTransformer
import re

def get_embeddings(texts, model):
    return np.array(model.encode(texts))

def preprocess(text):
    delimiters = r"[|–-]"
    split_text = re.split(delimiters, text)
    enough_info_first = len(split_text) > 0 and len(text) - len(split_text[0]) > 5
    is_potential_domain_info = len(split_text) > 1 and len(split_text[-1]) < 20
    if enough_info_first and is_potential_domain_info:
        return ' '.join(filter(None, split_text[:-1])).strip().lower()
    return text.lower()

@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

def reduce_dimensions(embeddings, method='PCA', n_components=2):
    """
    Reduce dimensionality of embeddings to 2D or 3D using PCA, t-SNE, or UMAP.
    """
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, learning_rate='auto', init='random',
                         perplexity=min(len(embeddings) - 1, 30))
    else:
        raise ValueError(f"Unknown method: {method}")
    reduced = reducer.fit_transform(embeddings)
    return reduced

def main():
    st.title("Hugging Face Embeddings Visualization")

    # --- Model Selection ---
    model_name = st.text_input(
        "Enter Hugging Face Model Name",
        value="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load the model
    model = load_model(model_name)

    # --- Text Input ---
    user_input = st.text_area(
        "Enter one or more sentences (each on a new line):",
        "Hello world!\nStreamlit is awesome.\nHugging Face rocks!"
    )
    texts = [preprocess(line.strip()) for line in user_input.split('\n') if line.strip()]

    # --- Reduction Options ---
    method = st.radio("Dimensionality Reduction Method", ("PCA", "t-SNE"))
    dims = st.radio("Number of Dimensions to Display", (2, 3))

    # --- Button to Compute ---
    if st.button("Compute & Visualize Embeddings"):
        if len(texts) == 0:
            st.warning("Please enter some text.")
        else:
            # Get embeddings
            embeddings = get_embeddings(texts, model)
            # Reduce to 2D or 3D
            reduced_embeddings = reduce_dimensions(embeddings, method=method, n_components=dims)

            if dims == 2:
                # 2D Plot
                fig = px.scatter(
                    x=reduced_embeddings[:, 0],
                    y=reduced_embeddings[:, 1],
                    text=texts,
                    title=f"{method} Embeddings (2D)"
                )
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 3D Plot
                fig = px.scatter_3d(
                    x=reduced_embeddings[:, 0],
                    y=reduced_embeddings[:, 1],
                    z=reduced_embeddings[:, 2],
                    text=texts,
                    title=f"{method} Embeddings (3D)"
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
