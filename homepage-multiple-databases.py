"""
This script loads the clusters and displays them, with other info including
a feature circuit implementing their behavior, in a Streamlit app.

We need to keep three variables in the st.session_stateE:
  1. db_option: the selected database (clustering method)
  2. selected_metric: the selected metric to rank the clusters
    - this will be a dictionary mapping db_option to the metric name. This way,
        when users switch back and forth between db_options, the selectric metric
        will be preserved.
  3. cluster rank: the index of the cluster in the ranking
    - this will be a dictionary mapping a (db_option, selected_metric) pair to the
        cluster rank. This way, when users switch between db_options and metrics,
        the cluster rank will be preserved.
"""

from collections import defaultdict
import pickle
import gzip
import io
from PIL import Image
from io import BytesIO
import os
import base64
import json

import numpy as np
import matplotlib.pyplot as plt

from sqlitedict import SqliteDict
import streamlit as st
from streamlit_shortcuts import add_keyboard_shortcuts

from render_utils import tokens_to_html


# these are the dbs (clustering methods)
VISIBLE_DATABASES = {
    "sae-features_lin-effects_final-1-pos_nsamples8192_nctx64": "SAE Features Linear Effects Final 1 Position",
    # "sae-features_lin-effects_final-1-pos_nsamples8192_nctx64": "SAE Features Linear Effects Final 5 Positions",
    "sae-features_lin-effects_sum-over-pos_nsamples8192_nctx64": "SAE Features Linear Effects Sum Over Position",
    "sae-features_activations_final-1-pos_nsamples8192_nctx64": "SAE Features Activations Final 1 Position",
    "sae-features_activations_final-5-pos_nsamples8192_nctx64": "SAE Features Activations Final 5 Positions",
    "sae-features_activations_sum-over-pos_nsamples8192_nctx64": "SAE Features Activations Sum Over Position",
    "parameter-gradient-projections": "Parameter Gradient Projections",
}

def format_db_name(db_name):
    return VISIBLE_DATABASES[db_name]

# these are the metrics, which we display, and can sort the clusters by
METRIC_DESCRIPTIONS = {
    "identity": {
        "title": "Identity",
        "description": "Clusters are ranked by their index."
    },
    "n_samples": {
        "title": "Number of samples",
        "description": "The total number of samples in the cluster."
    },
    "n_nodes": {
        "title": "Number of nodes",
        "description": "The total number of nodes (squares + triangles) in the circuit."
    },
    "n_triangles": {
        "title": "Number of triangles",
        "description": "The total number of triangles in the circuit."
    },
    "relative_max_feature_effect_node": {
        "title": "Relative max. feature effect (node)",
        "description": "max(abs(feature_effect)) / mean(abs(feature_effect))"
    },
    "relative_max_feature_effect_edge": {
        "title": "Relative max. feature effect (edge)",
        "description": "max(abs(feature_effect)) / mean(abs(feature_effect))"
    },
    "relative_softmaxx_feature_effects_node": {
        "title": "Relative Softmax",
        "description": "f(feature_effects) / (f(feature_effects) + f(error_effects)) for f(x) = sum(x * softmax(x))"
    },
    # "relative_writer_effect_node": {
    #     "title": "Relative writer effect (node)",
    #     "description": "sum(attn_features, mlp_features) / sum(attn_features, mlp_features, resid_features)"
    # },
    "None": {
        "title": "None",
        "description": "No sorting"
    }
}

def format_metric_name(metric):
    return METRIC_DESCRIPTIONS[metric]['title']

# first time user opens the app, load up all rankings
if 'metric_rankings' not in st.session_state:
    st.session_state['available_metrics'] = dict()
    for db_option in VISIBLE_DATABASES:
        with open(f"data/{db_option}/metrics.json") as f:
            st.session_state['available_metrics'][db_option] = json.load(f)


# Sidebar
st.sidebar.header('Cluster choice')

st.sidebar.selectbox('Select clustering method', 
                        list(VISIBLE_DATABASES.keys()),
                        format_func=format_db_name,
                        index=list(VISIBLE_DATABASES.keys()).index("sae-features_lin-effects_sum-over-pos_nsamples8192_nctx64"), 
                        key="db_option")
with open(f"data/{st.session_state['db_option']}/meta.json") as f:
    metadata = json.load(f)
with open(f"data/{st.session_state['db_option']}/metrics.json") as f:
    metric_options = json.load(f)
    new_dict = {}
    for metric in metric_options: # remove metrics that are not in METRIC_DESCRIPTIONS
        if metric in METRIC_DESCRIPTIONS:
            new_dict[metric] = metric_options[metric]
    metric_options = new_dict
    
metric_options['None'] = list(range(metadata['n_clusters']))
# choose default metric
if "relative_softmaxx_feature_effects_node" in metric_options:
    default_metric = "relative_softmaxx_feature_effects_node"
else:
    default_metric = "None"
st.sidebar.selectbox('Sort clusters by', 
                        list(metric_options),
                        format_func=format_metric_name,
                        index=list(metric_options).index(default_metric),
                        key='metric')
st.sidebar.selectbox('Select cluster rank', 
                        range(metadata['n_clusters']), 
                        index=metadata['starting_cluster_idx'], 
                        key='cluster_rank')

def increment_cluster_rank():
    rank = st.session_state['cluster_rank']
    next_rank = min(rank + 1, int(metadata['n_clusters']) - 1)
    st.session_state['cluster_rank'] = next_rank

def decrement_cluster_rank():
    rank = st.session_state['cluster_rank']
    next_rank = max(rank - 1, 0)
    st.session_state['cluster_rank'] = next_rank
    
if st.sidebar.button('Previous cluster', on_click=decrement_cluster_rank):
    pass
if st.sidebar.button('Next cluster', on_click=increment_cluster_rank):
    pass

# add keyboard shortcuts
add_keyboard_shortcuts({
    "ArrowLeft": "Previous cluster",
    "ArrowRight": "Next cluster"
})

# add text to the sidebar
st.sidebar.write(f"You can use the left and right arrow keys to move quickly between clusters.")

st.sidebar.write("**Browse features on neuronpedia:** https://www.neuronpedia.org/p70d-sm")

# also show the database description
st.sidebar.write(metadata['database_description'])

# Now for the main page
clusteri = metric_options[st.session_state['metric']][st.session_state['cluster_rank']]

st.write(f"## Cluster #{clusteri}")
st.write(f"(Ranked {st.session_state['cluster_rank']}/{int(metadata['n_clusters'])} using the {format_metric_name(st.session_state['metric'])} metric)")

# load up the cluster data
with SqliteDict(f"data/{st.session_state['db_option']}/database.sqlite") as db:
    compressed_bytes = db[clusteri]
    decompressed_object = io.BytesIO(compressed_bytes)
    with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
        cluster_data = pickle.load(file)
if os.path.exists(f"data/{st.session_state['db_option']}/circuit_images.sqlite"):
    with SqliteDict(f"data/{st.session_state['db_option']}/circuit_images.sqlite") as db:
        compressed_bytes = db[clusteri]
        decompressed_object = io.BytesIO(compressed_bytes)
        with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
            cluster_data.update(pickle.load(file))
if os.path.exists(f"data/{st.session_state['db_option']}/circuit_graphviz.sqlite"):
    with SqliteDict(f"data/{st.session_state['db_option']}/circuit_graphviz.sqlite") as db:
        compressed_bytes = db[clusteri]
        decompressed_object = io.BytesIO(compressed_bytes)
        with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
            cluster_data.update(pickle.load(file))

# Add a download button for the contexts in the sidebar
st.sidebar.write("Download the contexts for this cluster:")
st.sidebar.download_button(
    label="Download contexts",
    data=json.dumps(cluster_data['contexts']),
    file_name=f"cluster_{clusteri}_contexts.json",
    mime="application/json"
)

if 'circuit_image' in cluster_data and cluster_data['circuit_image'] is not None:
    st.sidebar.write("Download the circuit image for this cluster:")
    if isinstance(cluster_data['circuit_image'], Image.Image):
        buffer = io.BytesIO()
        # save to buffer, max quality
        cluster_data['circuit_image'].save(buffer, format='PNG', quality=400)
        cluster_data['circuit_image'] = buffer.getvalue()
    st.sidebar.download_button(
        label="Download circuit image (high res)",
        data=cluster_data['circuit_image'],
        file_name=f"cluster_{clusteri}_circuit_image.png",
        mime="image/png",
    )

st.markdown("### Summary statistics")


def get_mean_loss():
    if 'mean_loss' not in st.session_state:
        st.session_state['mean_loss'] = np.load(f"data/mean_loss_curve.npy")
        return st.session_state['mean_loss']
    else:
        return st.session_state['mean_loss']

# Create a single figure with subplots
fig = plt.figure(figsize=(8, 6))

# Subplot 0: Last token frequencies
ax0 = plt.subplot(2, 2, 1)
counts = defaultdict(int)
for context in cluster_data['contexts'].values():
    y = context['context'][-1]
    counts[y] += 1

top_10 = sorted(counts, key=counts.get, reverse=True)[:10]
top_10_counts = [counts[y] for y in top_10]
top_10 = [repr(y)[1:-1] for y in top_10]
ax0.bar(top_10, top_10_counts)
ax0.set_xlabel('Token', fontsize=12)
ax0.set_ylabel('Count', fontsize=12)
ax0.set_title('Last token frequencies', fontsize=12)
ax0.tick_params(axis='x', rotation=45, labelsize=10)
ax0.tick_params(axis='y', labelsize=10)

# Subplot 1: Next token frequencies
ax1 = plt.subplot(2, 2, 2)
counts = defaultdict(int)
for context in cluster_data['contexts'].values():
    y = context['answer']
    counts[y] += 1

top_10 = sorted(counts, key=counts.get, reverse=True)[:10]
top_10_counts = [counts[y] for y in top_10]
top_10 = [repr(y)[1:-1] for y in top_10]
ax1.bar(top_10, top_10_counts)
ax1.set_xlabel('Token', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Next token frequencies', fontsize=12)
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

if 'circuit_graphviz' in cluster_data and cluster_data['circuit_graphviz'] is not None:
    st.graphviz_chart(cluster_data['circuit_graphviz'])
elif 'circuit_image' in cluster_data and cluster_data['circuit_image'] is not None: # Display the circuit image
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
if 'circuit_metrics' in cluster_data and cluster_data['circuit_metrics'] is not None:
    st.write("")
    # redundant due to missing data
    if 'n_samples' not in METRIC_DESCRIPTIONS:
        st.markdown(f"**Number of samples**: {len(cluster_data['contexts'])}", help=METRIC_DESCRIPTIONS['n_samples']['description'])

    for metric_name in METRIC_DESCRIPTIONS:
        if metric_name not in cluster_data['circuit_metrics']:
            continue
        title = METRIC_DESCRIPTIONS[metric_name]['title']
        desc = METRIC_DESCRIPTIONS[metric_name]['description']
        number = cluster_data['circuit_metrics'][metric_name]
        if isinstance(number, float):
            number = np.round(number, 3)
        st.markdown(f"**{title}**: {number}", help=desc)

# Display the contexts
for context in cluster_data['contexts'].values():
    y = context['answer']
    tokens = context['context'] + [y]
    html = tokens_to_html(tokens)
    st.write("-----------------------------------------------------------")
    st.write(html, unsafe_allow_html=True)

