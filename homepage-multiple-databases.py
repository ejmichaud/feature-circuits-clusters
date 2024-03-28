
"""
This script loads the clusters and displays them, with other info including
a feature circuit implementing their behavior, in a Streamlit app.
"""

from collections import defaultdict
import pickle
import gzip
import io
from PIL import Image
from io import BytesIO
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from sqlitedict import SqliteDict
import streamlit as st
from streamlit_shortcuts import add_keyboard_shortcuts

from render_utils import tokens_to_html


#####################
# Constants
#####################

st.session_state['db_option'] = st.session_state.get('db_option', None)
st.session_state['selected_metric_rank'] = st.session_state.get('selected_metric_rank', "Identity")
st.session_state['circuit_images_path'] = st.session_state.get('circuit_images_path', None)
st.session_state['metric_ranks'] = st.session_state.get('metric_ranks', dict())
st.session_state['rank_to_name'] = st.session_state.get('rank_to_name', None)
st.session_state['name_to_rank'] = st.session_state.get('name_to_rank', None)
st.session_state['cluster_name'] = st.session_state.get('cluster_name', None)
st.session_state['n_clusters'] = st.session_state.get('n_clusters', None)
st.session_state['metric_descriptions'] = st.session_state.get('metric_descriptions', dict(
    identity = dict(title="Identity", description="Clusters are ranked by their index."),
    n_samples = dict(title="Number of samples", description="The total number of samples in the cluster."),
    n_nodes = dict(title="Number of nodes", description="The total number of nodes (squares + triangles) in the circuit."),
    n_triangles = dict(title="Number of triangles", description="The total number of triangles in the circuit."),
    # relative_max_feature_effect_node = dict(title="Relative max. feature effect (node)", description="max(abs(feature_effect)) / mean(abs(feature_effect))"),
    # relative_max_feature_effect_edge = dict(title="Relative max. feature effect (edge)", description="max(abs(feature_effect)) / mean(abs(feature_effect))"),
    # relative_writer_effect_node = dict(title="Relative writer effect (node)", description="sum(attn_features, mlp_features) / sum(attn_features, mlp_features, resid_features)"),
    relative_softmaxx_feature_effects_node = dict(title="Relative Softmax", description="f(feature_effects) / (f(feature_effects) + f(error_effects)) for f(x) = sum(x * softmax(x))"),
))
st.session_state['metric_title_to_name'] = st.session_state.get('metric_title_to_name', {v['title']: k for k, v in st.session_state['metric_descriptions'].items()})

VISIBLE_DATABASES = {
    "sae-features_lin-effects_final-1-pos_nsamples8192_nctx64": "SAE Features Linear Effects Final 1 Position",
    # "sae-features_lin-effects_final-1-pos_nsamples8192_nctx64": "SAE Features Linear Effects Final 5 Positions",
    "sae-features_lin-effects_sum-over-pos_nsamples8192_nctx64": "SAE Features Linear Effects Sum Over Position",
    "sae-features_activations_final-1-pos_nsamples8192_nctx64": "SAE Features Activations Final 1 Position",
    "sae-features_activations_final-5-pos_nsamples8192_nctx64": "SAE Features Activations Final 5 Positions",
    "sae-features_activations_sum-over-pos_nsamples8192_nctx64": "SAE Features Activations Sum Over Position",
    "parameter-gradient-projections": "Parameter Gradient Projections",
}
VISIBLE_DATABASES_TITLE_TO_NAME = {v: k for k, v in VISIBLE_DATABASES.items()}


#####################
# Database & helper function initialization
#####################

def load_database():
    selected_db = VISIBLE_DATABASES_TITLE_TO_NAME[st.session_state['db_option']]

    with open(f"data/{selected_db}/meta.json") as f:
        metadata = json.load(f)
    st.session_state['n_clusters'] = metadata['n_clusters']
    st.session_state['cluster_name'] = metadata['starting_cluster_idx']
    # print("just set cluster name to", st.session_state['cluster_name'])
    st.session_state['database_description'] = metadata['database_description']
    st.session_state['selected_metric_rank'] = "Identity"

    # If statements due to inconsistent database formats
    database_filenames = [f for f in os.listdir(f"data/{selected_db}")]
    if "circuit_images.sqlite" in database_filenames:
        st.session_state['circuit_images_path'] = f"data/{selected_db}/circuit_images.sqlite"
    if "metrics.json" in database_filenames:
        with open(f"data/{selected_db}/metrics.json") as f:
            st.session_state['metric_ranks'] = json.load(f)
    st.session_state['metric_ranks']['identity'] = np.arange(st.session_state['n_clusters'])

def assign_metric():
    selected_metric_rank = st.session_state['selected_metric_rank']
    selected_metric_rank = st.session_state['metric_title_to_name'][selected_metric_rank]
    st.session_state['rank_to_name'] = st.session_state['metric_ranks'][selected_metric_rank]
    # Map values to indices
    st.session_state['name_to_rank'] = [-1] * len(st.session_state['rank_to_name'])
    for index, value in enumerate(st.session_state['rank_to_name']):
        st.session_state['name_to_rank'][value] = index

    # Set the cluster name to the first cluster in the ranking
    if st.session_state['selected_metric_rank'] != 'Identity':
        st.session_state['cluster_name'] = st.session_state['rank_to_name'][0]

# Load database on startup
database_names = sorted([f for f in os.listdir("data") if os.path.isdir(os.path.join("data", f))])
if st.session_state['db_option'] is None:
    st.session_state['db_option'] = VISIBLE_DATABASES[database_names[0]]
    selected_db = VISIBLE_DATABASES_TITLE_TO_NAME[st.session_state['db_option']]
    load_database()
    assign_metric()


# Initialize the helper functions
    
# Only called if "losses" in dataset
def get_mean_loss():
    if 'mean_loss' not in st.session_state:
        st.session_state['mean_loss'] = np.load(f"data/mean_loss_curve.npy")
        return st.session_state['mean_loss']
    else:
        return st.session_state['mean_loss']
    
#     def get_idxs():
#         if 'idxs' not in st.session_state:
#             with open(idxs_path, "rb") as f:
#                 st.session_state['idxs'] = pickle.load(f)
#             return st.session_state['idxs']
#         else:
#             return st.session_state['idxs']

# def filter_clusters(search_query):
#     if search_query:
#         return [idx for idx in range(int(st.session_state['n_clusters'])) if search_query.lower() in str(idx)]
#     else:
#         return range(int(st.session_state['n_clusters']))


#####################
# Sidebar
#####################

# Create sidebar for selecting clusters file and cluster
st.sidebar.header('Cluster choice')

# Selectbox for one of the clusters in the data directory
st.sidebar.selectbox('Select clustering method', list(VISIBLE_DATABASES.values()), key="db_option", on_change=load_database)
selected_db = VISIBLE_DATABASES_TITLE_TO_NAME[st.session_state['db_option']]

# Select ranking metric
if st.session_state['metric_ranks'] is not None:
    metric_options = [st.session_state['metric_descriptions'][m]['title'] for m in st.session_state['metric_descriptions'] if m in st.session_state['metric_ranks']]
else:
    metric_options = ['identity']
st.sidebar.selectbox('Sort clusters by', metric_options, index=0, key='selected_metric_rank', on_change=assign_metric)

## Cluster Rank (The rank does not map to a unique cluster. It is the index to the clusters ranked by a metric chosen below.)
def get_cluster_name():
    return st.session_state['cluster_name']

def get_cluster_rank():
    c = st.session_state['cluster_name']
    return int(st.session_state['name_to_rank'][c])

def set_cluster_rank(i):
    st.session_state['cluster_name'] = st.session_state['rank_to_name'][i]
    return st.session_state['cluster_name']
        
def increment_cluster_rank():
    rank = get_cluster_rank()
    next_rank = min(rank + 1, int(st.session_state['n_clusters']) - 1)
    return set_cluster_rank(next_rank)

def decrement_cluster_rank():
    rank = get_cluster_rank()
    next_rank = max(rank - 1, 0)
    return set_cluster_rank(next_rank)
    

# def get_cluster_name_str():
#     """Converts a cluster name to a string to avoid confusion with the cluster rank"""
#     return "".join([chr(97 + int(c)) for c in str(st.session_state['cluster_name'])]).upper()

# Choose a cluster rank
cluster_rank = st.sidebar.selectbox('Select cluster rank', range(int(st.session_state['n_clusters'])), index=get_cluster_rank())
set_cluster_rank(cluster_rank)

def left_callback():
    decrement_cluster_rank()

def right_callback():
    increment_cluster_rank()

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


st.write(f"## Cluster #{get_cluster_name()}")
st.write(f'Cluster #{st.session_state["cluster_name"]} is ranked {get_cluster_rank()} out of {int(st.session_state["n_clusters"])} using the metric "{st.session_state["selected_metric_rank"]}".')

with SqliteDict(f"data/{selected_db}/database.sqlite") as db:
    compressed_bytes = db[str(get_cluster_name())]
    decompressed_object = io.BytesIO(compressed_bytes)
    with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
        cluster_data = pickle.load(file)


# Separate circuit image database for quick iterations, remove later for faster access
if st.session_state['circuit_images_path'] is not None:
    with SqliteDict(st.session_state['circuit_images_path']) as db:
        compressed_bytes = db[str(get_cluster_name())]
        decompressed_object = io.BytesIO(compressed_bytes)
        with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
            cluster_data['circuit_image'] = pickle.load(file)['circuit_image']


# Add a download button for the contexts in the sidebar
contexts = cluster_data['contexts']
st.sidebar.write("Download the contexts for this cluster:")
st.sidebar.download_button(
    label="Download contexts",
    data=pickle.dumps(contexts),
    file_name=f"cluster_{get_cluster_name()}_contexts.pkl",
    mime="application/octet-stream"
)

# if 'circuit_image' in cluster_data:
#     st.sidebar.write("Download the circuit image for this cluster:")
#     if isinstance(cluster_data['circuit_image'], Image.Image):
#         buffer = io.BytesIO()
#         cluster_data['circuit_image'].save(buffer, format='PNG')
#         cluster_data['circuit_image'] = buffer.getvalue()
#     st.sidebar.download_button(
#         label="Download circuit image (high res)",
#         data=cluster_data['circuit_image'],
#         file_name=f"cluster_{get_cluster_name()}_circuit_image.png",
#         mime="image/png",
#     )

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
if 'circuit_image' in cluster_data:
    if cluster_data['circuit_image'] is not None:
        if isinstance(cluster_data['circuit_image'], Image.Image):
            img = cluster_data['circuit_image']
        elif isinstance(cluster_data['circuit_image'], bytes):
            img = Image.open(BytesIO(cluster_data['circuit_image']))
        else:
            raise ValueError(f"Unexpected type for circuit image: {type(cluster_data['circuit_image'])}")
        st.image(img, use_column_width=None, output_format='PNG')
    else:
        st.write("No circuit image available.")

# Display metrics for the cluster circuit
if 'circuit_metrics' in cluster_data:
    st.write("")
    # redundant due to missing data
    if 'n_samples' not in st.session_state["metric_descriptions"]:
        st.markdown(f"**Number of samples**: {len(contexts)}", help=st.session_state["metric_descriptions"]['n_samples']['description'])

    for metric_name in st.session_state["metric_descriptions"]:
        if metric_name not in cluster_data['circuit_metrics']:
            continue
        title = st.session_state["metric_descriptions"][metric_name]['title']
        desc = st.session_state["metric_descriptions"][metric_name]['description']
        number = cluster_data['circuit_metrics'][metric_name]
        if isinstance(number, float):
            number = np.round(number, 3)
        st.markdown(f"**{title}**: {number}", help=desc)

# Display the contexts
for context in contexts.values():
    y = context['answer']
    tokens = context['context'] + [y]
    html = tokens_to_html(tokens)
    st.write("-----------------------------------------------------------")
    st.write(html, unsafe_allow_html=True)
