
"""
This script loads the clusters and corresponding contexts from the data folder and
displays them in a Streamlit app. 

This displays the data with a loss thresholdhold of 0.3.
Uses pythia-70m-deduped now.
"""

from collections import defaultdict
import pickle
import gzip
import io
from sqlitedict import SqliteDict
from PIL import Image
from io import BytesIO
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_shortcuts import add_keyboard_shortcuts

from render_utils import tokens_to_html


# Create sidebar for selecting clusters file and cluster
st.sidebar.header('Cluster choice')

# Selectbox for one of the clusters in the data directory
database_names = [f for f in os.listdir("data") if os.path.isdir(os.path.join("data", f))]
selected_database = st.sidebar.selectbox('Select clustering parameters', database_names)
database_filenames = [f for f in os.listdir(f"data/{selected_database}")]

idxs_path, mean_loss_curve_path = None, None
if "idxs.pkl" in database_filenames:
    idxs_path = f"data/{selected_database}/idxs.pkl"
if "mean_loss_curve.npy" in database_filenames:
    mean_loss_curve_path = f"data/{selected_database}/mean_loss_curve.npy"

if idxs_path:
    def get_idxs():
        if 'idxs' not in st.session_state:
            with open(idxs_path, "rb") as f:
                st.session_state['idxs'] = pickle.load(f)
            return st.session_state['idxs']
        else:
            return st.session_state['idxs']
        
if mean_loss_curve_path:
    def get_mean_loss():
        if 'mean_loss' not in st.session_state:
            st.session_state['mean_loss'] = np.load(mean_loss_curve_path)
            return st.session_state['mean_loss']
        else:
            return st.session_state['mean_loss']
    
def filter_clusters(search_query):
    if search_query:
        return [idx for idx in range(int(st.session_state['n_clusters'])) if search_query.lower() in str(idx)]
    else:
        return range(int(st.session_state['n_clusters']))


# sort clusters by size (dictionary of rank -> old cluster index)
# new_index_old_index = {i: cluster for i, cluster in enumerate(sorted(cluster_to_tokens, key=lambda k: len(cluster_to_tokens[k]), reverse=True))}

# (Re)initialize session state on database change
st.session_state['selected_database'] = st.session_state.get('selected_database', None)
if st.session_state['selected_database'] != selected_database:
    st.session_state['selected_database'] = selected_database
    # Read metadata
    with open(f"data/{selected_database}/meta.json") as f:
        metadata = json.load(f)
    st.session_state['n_clusters'] = st.session_state.get('n_clusters', metadata['n_clusters'])
    st.session_state['clusteri'] = metadata['starting_cluster_idx']
    st.session_state['database_description'] = metadata['database_description']

def get_clusteri():
    return st.session_state['clusteri']

def set_clusteri(i):
    st.session_state['clusteri'] = i
    return st.session_state['clusteri']

def increment_clusteri():
    st.session_state['clusteri'] = get_clusteri() + 1
    return get_clusteri()

def decrement_clusteri():
    st.session_state['clusteri'] = get_clusteri() - 1
    return get_clusteri()


# choose a cluster index
cluster_idx = st.sidebar.selectbox('Select cluster index', range(int(st.session_state['n_clusters'])), index=get_clusteri())
set_clusteri(cluster_idx)

def left_callback():
    decrement_clusteri()

def right_callback():
    increment_clusteri()

# these don't take any action. fix this:
if st.sidebar.button('Previous cluster', on_click=left_callback):
    pass
if st.sidebar.button('Next cluster', on_click=right_callback):
    pass

# add keyboard shortcuts
add_keyboard_shortcuts({
    "ArrowLeft": "Previous cluster",
    "ArrowRight": "Next cluster"
})

# add text to the sidebar
st.sidebar.write(f"You can use the left and right arrow keys to move quickly between clusters.")

st.sidebar.write(st.session_state['database_description'])

# load up the contexts and the clusters
# if "eric" in cluster_file.lower():
#     with open("data/ERIC-QUANTA-CONTEXTS.json") as f:
#         samples = json.load(f)
# else:
#     with open("data/contexts_pythia-70m-deduped_tloss0.03_ntok10000_skip512_npos10_mlp.json") as f:
#         samples = json.load(f)
# if 'samples' not in st.session_state:
#     with open("data-large008/contexts-pythia-70m-100k.json") as f:
#         # samples = json.load(f)
#         st.session_state['samples'] = json.load(f)

# idx_to_token_idx = list(get_contexts().keys())
# ...

# write as large bolded heading the cluster index
st.write(f"## Cluster {get_clusteri()}")

# load up the cluster data from the database
if "database_stats.sqlite" in database_filenames:
    database_path = f"data/{selected_database}/database_stats.sqlite"
else:
    database_path = f"data/{selected_database}/database.sqlite"

with SqliteDict(database_path) as db:
    compressed_bytes = db[get_clusteri()]
    decompressed_object = io.BytesIO(compressed_bytes)
    with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
        cluster_data = pickle.load(file)

# Separate circuit image database for quick iterations, remove later for faster access
if "circuit_images.sqlite" in database_filenames:
    with SqliteDict(f"data/{selected_database}/circuit_images.sqlite") as db:
        compressed_bytes = db[get_clusteri()]
        decompressed_object = io.BytesIO(compressed_bytes)
        with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
            cluster_data['circuit_image'] = pickle.load(file)['circuit_image']



# Add a download button for the contexts in the sidebar
contexts = cluster_data['contexts']
st.sidebar.write("Download the contexts for this cluster:")
st.sidebar.download_button(
    label="Download contexts",
    data=pickle.dumps(contexts),
    file_name=f"cluster_{get_clusteri()}_contexts.pkl",
    mime="application/octet-stream"
)

# Create a single figure with subplots
fig = plt.figure(figsize=(8, 6))

# Subplot 1: Histogram of top 10 tokens
ax1 = plt.subplot(2, 2, (1, 2))
counts = defaultdict(int)
for context in contexts.values():
    y = context['answer']
    counts[y] += 1

top_10 = sorted(counts, key=counts.get, reverse=True)[:10]
top_10_counts = [counts[y] for y in top_10]
top_10 = [repr(y)[1:-1] for y in top_10]
ax1.bar(top_10, top_10_counts)
ax1.set_xlabel('Token', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Top 10 tokens in cluster (answer tokens)', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.tick_params(axis='y', labelsize=10)

# Subplot 2: Permuted Cs
if 'permuted_C' in cluster_data:
    permuted_C = cluster_data['permuted_C']
    ax2 = plt.subplot(2, 2, 3)
    im2 = ax2.imshow(permuted_C, cmap='rainbow', vmin=-1, vmax=1)
    ax2.set_title('Similarity matrix for cluster', fontsize=10)
    # ticks for the axes should be integers. there should be 5 of them
    ticks = list(range(0, permuted_C.shape[0], max(1, permuted_C.shape[0] // 5)))
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    plt.colorbar(im2, ax=ax2)

# Subplot 3: Loss curves
if 'losses' in cluster_data:
    losses = cluster_data['losses']
    ax3 = plt.subplot(2, 2, 4)
    steps = [0] + [2**i for i in range(10)] + list(range(1000, 144000, 1000))
    for i in range(len(losses)):
        ax3.plot(steps, losses[i], color='black', alpha=0.2)
    ax3.plot(steps, get_mean_loss(), label="mean loss", color='red')
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Loss curves for tokens in cluster', fontsize=10)
    ax3.set_xscale('log')
    # add tick marks for every 10^i
    ax3.set_xticks([10**i for i in range(5)])
    ax3.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Display the figure using st.pyplot()
st.pyplot(fig)

# Display the circuit image
if cluster_data['circuit_image'] is not None:
    if isinstance(cluster_data['circuit_image'], Image.Image):
        img = cluster_data['circuit_image']
    elif isinstance(cluster_data['circuit_image'], bytes):
        img = Image.open(BytesIO(cluster_data['circuit_image']))
    else:
        raise ValueError(f"Unexpected type for circuit image: {type(cluster_data['circuit_image'])}")
    st.image(img, use_column_width=None, output_format='PNG')


metric_descriptions = dict(
    n_nodes = dict(title="Number of nodes", description="The total number of nodes (squares + triangles) in the circuit."),
    n_triangles = dict(title="Number of triangles", description="The total number of triangles in the circuit."),
    relative_max_feature_effect_node = dict(title="Relative max. feature effect (node)", description="max(abs(feature_effect)) / mean(abs(feature_effect))"),
    relative_max_feature_effect_edge = dict(title="Relative max. feature effect (edge)", description="max(abs(feature_effect)) / mean(abs(feature_effect))"),
    relative_writer_effect_node = dict(title="Relative writer effect (node)", description="sum(attn_features, mlp_features) / sum(attn_features, mlp_features, resid_features)"),
    relative_softmaxx_feature_effects_node = dict(title="Aaron & Sam Interestingness", description="f(feature_effects) / (f(feature_effects) + f(error_effects)) for f(x) = sum(x * softmax(x))"),
)

# Display metrics for the cluster circuit
if 'circuit_metrics' in cluster_data:
    st.write("")
    for metric_name in cluster_data['circuit_metrics']:
        title = metric_descriptions[metric_name]['title']
        desc = metric_descriptions[metric_name]['description']
        number = cluster_data['circuit_metrics'][metric_name]
        if isinstance(number, float):
            number = np.round(number, 3)
        st.markdown(f"**{title}**: {number}", help=desc)


for context in contexts.values():
    y = context['answer']
    tokens = context['context'] + [y]
    html = tokens_to_html(tokens)
    st.write("-----------------------------------------------------------")
    st.write(html, unsafe_allow_html=True)