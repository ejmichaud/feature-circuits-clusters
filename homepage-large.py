"""
This script loads the clusters and corresponding contexts from the data folder and
displays them in a Streamlit app. 
"""

from collections import defaultdict
import json
import pickle
import gzip

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_shortcuts import add_keyboard_shortcuts

def tokens_to_html(tokens, max_len=150):
    """Given a list of tokens (strings), returns html for displaying the tokenized text.
    """
    newline_tokens = ['\n', '\r', '\r\n', '\v', '\f']
    html = ""
    txt = ""
    if len(tokens) > max_len:
        html += '<span>...</span>'
    tokens = tokens[-max_len:]
    for i, token in enumerate(tokens):
        background_color = "white" if i != len(tokens) - 1 else "#FF9999"
        txt += token
        if all([c in newline_tokens for c in token]):
            # replace all instances with ⏎
            token_rep = len(token) * "⏎"
            brs = "<br>" * len(token)
            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; white-space: pre-wrap;">{token_rep}</span>{brs}'
        else:
            # replace any $ with \$ to avoid markdown interpretation
            token = token.replace("$", "\$")
            # replace any < with &lt; to avoid html interpretation
            # token = token.replace("<", "&lt;")
            # replace any > with &gt; to avoid html interpretation
            # token = token.replace(">", "&gt;")
            # replace any & with &amp; to avoid html interpretation
            token = token.replace("&", "&amp;")
            # replace any _ with \_ to avoid markdown interpretation
            token = token.replace("_", "\_")
            # also escape * to avoid markdown interpretation
            token = token.replace("*", "\*")
            # there's also an issue with the backtick, so escape it
            token = token.replace("`", "\`")

            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; white-space: pre-wrap;">{token}</span>'
    if "</" in txt:
        return "CONTEXT NOT LOADED FOR SECURITY REASONS SINCE IT CONTAINS HTML CODE (could contain javascript)."
    else:
        return html


# Create sidebar for selecting clusters file and cluster
st.sidebar.header('Cluster choice')

# Selectbox for the clusters file
# cluster_files = os.listdir("data-large")
# cluster_files.remove("contexts-pythia-70m-100k.json")
# # cluster_files.remove("contexts_pythia-70m-deduped_tloss0.03_ntok10000_skip512_npos10_mlp.json")
# # cluster_files.remove("ERIC-QUANTA-CONTEXTS.json")
# cluster_file = st.sidebar.selectbox('Select cluster file', cluster_files)
# with open(f"data-large/{cluster_file}") as f:
#     clusters = json.load(f)

def get_clusters():
    if 'clusters' not in st.session_state:
        with open("data-large/cluster_is.pkl", "rb") as f:
            st.session_state['clusters'] = pickle.load(f)
            return st.session_state['clusters']
    else:
        return st.session_state['clusters']

def get_contexts():
    if 'samples' not in st.session_state:
        # demonstrating loading of this saved file .json.gz now
        with gzip.open("data-large/contexts-pythia-70m-100k.json.gz", "rb") as f:
            st.session_state['samples'] = pickle.load(f)
        return st.session_state['samples']
    else:
        return st.session_state['samples']

def get_idxs():
    if 'idxs' not in st.session_state:
        with open("data-large/idxs.pkl", "rb") as f:
            st.session_state['idxs'] = pickle.load(f)
        return st.session_state['idxs']
    else:
        return st.session_state['idxs']
    
def get_losses():
    if 'losses' not in st.session_state:
        st.session_state['losses'] = np.load("data-large/loss_curves.npy")
        return st.session_state['losses']
    else:
        return st.session_state['losses']

def get_mean_loss():
    if 'mean_loss' not in st.session_state:
        st.session_state['mean_loss'] = np.load("data-large/mean_loss_curve.npy")
        return st.session_state['mean_loss']
    else:
        return st.session_state['mean_loss']

def get_permuted_Cs():
    if 'permuted_Cs' not in st.session_state:
        with gzip.open("data-large/permuted_Cs.pkl.gz", "rb") as f:
            st.session_state['permuted_Cs'] = pickle.load(f)
        return st.session_state['permuted_Cs']
    else:
        return st.session_state['permuted_Cs']
    
def get_unpermuted_Cs():
    if 'unpermuted_Cs' not in st.session_state:
        with open("data-large/unpermuted_Cs.pkl", "rb") as f:
            st.session_state['unpermuted_Cs'] = pickle.load(f)
        return st.session_state['unpermuted_Cs']
    else:
        return st.session_state['unpermuted_Cs']

# Selectbox for choosing n_clusters
# n_clusters_options = sorted(list(clusters.keys()), key=int)
# n_clusters = st.sidebar.selectbox('n_clusters used in clustering algorithm', n_clusters_options, index=len(n_clusters_options) - 1)
# clusters = clusters[n_clusters] # note that n_clusters is a string
# clusters = clusters[0] # ignore clusters[1], which is based on absolute values

# From the clusters list, create a dictionary mapping cluster index to token indices
# cluster_to_tokens = defaultdict(list)
# for i, cluster in enumerate(clusters):
#     cluster_to_tokens[cluster].append(i)

n_clusters = len(get_clusters())

# sort clusters by size (dictionary of rank -> old cluster index)
# new_index_old_index = {i: cluster for i, cluster in enumerate(sorted(cluster_to_tokens, key=lambda k: len(cluster_to_tokens[k]), reverse=True))}

def get_clusteri():
    if 'clusteri' not in st.session_state:
        st.session_state['clusteri'] = n_clusters // 3
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

# def get_idx(cluster_file, n_clusters):
#     if cluster_file not in st.session_state:
#         st.session_state[cluster_file] = dict()
#     if n_clusters not in st.session_state[cluster_file]:
#         st.session_state[cluster_file][n_clusters] = int(n_clusters) // 2
#     return st.session_state[cluster_file][n_clusters]

# def increment_idx(cluster_file, n_clusters):
#     st.session_state[cluster_file][n_clusters] += 1
#     return st.session_state[cluster_file][n_clusters]

# def decrement_idx(cluster_file, n_clusters):
#     st.session_state[cluster_file][n_clusters] -= 1
#     return st.session_state[cluster_file][n_clusters]

# def set_idx(cluster_file, n_clusters, idx):
#     st.session_state[cluster_file][n_clusters] = idx
#     return st.session_state[cluster_file][n_clusters]

# choose a cluster index
cluster_idx = st.sidebar.selectbox('Select cluster index', range(int(n_clusters)), index=get_clusteri())
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

st.sidebar.write("These are clusters for pythia-70m, with a loss threshold of 0.1 nats, across 100k tokens, using k-means for clustering. I projected the gradients (~19m-dimensional) onto a 30k-dimensional subspace for clustering. There are 2500 clusters in total.")

# load up the contexts and the clusters
# if "eric" in cluster_file.lower():
#     with open("data/ERIC-QUANTA-CONTEXTS.json") as f:
#         samples = json.load(f)
# else:
#     with open("data/contexts_pythia-70m-deduped_tloss0.03_ntok10000_skip512_npos10_mlp.json") as f:
#         samples = json.load(f)
# if 'samples' not in st.session_state:
#     with open("data-large/contexts-pythia-70m-100k.json") as f:
#         # samples = json.load(f)
#         st.session_state['samples'] = json.load(f)

# idx_to_token_idx = list(get_contexts().keys())
# ...

# write as large bolded heading the cluster index
st.write(f"## Cluster {get_clusteri()}")

# Create a single figure with subplots
fig = plt.figure(figsize=(8, 6))

# Subplot 1: Histogram of top 10 tokens
ax1 = plt.subplot(2, 2, (1, 2))
counts = defaultdict(int)
for i in get_clusters()[get_clusteri()]:
    idx = str(get_idxs()[i]) # the index into the pile tokens
    sample = get_contexts()[idx]
    y = sample['y']
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
ax2 = plt.subplot(2, 2, 3)
im2 = ax2.imshow(get_permuted_Cs()[get_clusteri()], cmap='rainbow', vmin=-1, vmax=1)
ax2.set_title('Similarity matrix for cluster', fontsize=10)
plt.colorbar(im2, ax=ax2)

# Subplot 3: Loss curves
ax3 = plt.subplot(2, 2, 4)
steps = [0] + [2**i for i in range(10)] + list(range(1000, 144000, 1000))
ax3.plot(steps, get_mean_loss(), label="mean loss", color='red')
for i in get_clusters()[get_clusteri()]:
    ax3.plot(steps, get_losses()[i], color='black', alpha=0.2)
ax3.set_xlabel('Step', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Loss curves for tokens in cluster', fontsize=10)
ax3.set_xscale('log')
ax3.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Display the figure using st.pyplot()
st.pyplot(fig)

# st.write(f"## Cluster {get_clusteri()}")

# plt.figure(figsize=(6, 6))

# plt.subplot(2, 2, 1)
# # get a histogram of the top 'y' tokens in the cluster
# counts = defaultdict(int)
# for i in get_clusters()[get_clusteri()]:
#     idx = str(get_idxs()[i]) # the index into the pile tokens
#     sample = get_contexts()[idx]
#     y = sample['y']
#     counts[y] += 1

# # plot the histogram for the top 10 tokens with matplotlib
# top_10 = sorted(counts, key=counts.get, reverse=True)[:10]
# top_10_counts = [counts[y] for y in top_10]
# # convert the top 10 tokens to literals (i.e. newlines and tabs are escaped)
# top_10 = [repr(y)[1:-1] for y in top_10]
# plt.figure(figsize=(6, 2))
# plt.bar(top_10, top_10_counts)
# plt.xlabel('Token', fontsize=8)
# plt.ylabel('Count', fontsize=8)
# plt.title('Top 10 tokens in cluster (answer tokens)', fontsize=8)
# # rotate the tick labels
# plt.xticks(rotation=45, fontsize=9)
# plt.yticks(fontsize=8)

# plt.subplot(2, 2, 2)
# plt.imshow(get_unpermuted_Cs()[get_clusteri()], cmap='viridis', vmin=-1, vmax=1)
# # print(get_unpermuted_Cs()[get_clusteri()].shape)
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.imshow(get_permuted_Cs()[get_clusteri()], cmap='viridis', vmin=-1, vmax=1)
# plt.colorbar()

# plt.subplot(2, 2, 4)
# # plot the mean loss curve and the loss curves for the elements in the cluster
# steps = [0] + [2**i for i in range(10)] + list(range(1000, 144000, 1000))

# plt.plot(steps, get_mean_loss(), label="mean loss", color='red')
# for i in get_clusters()[get_clusteri()]:
#     plt.plot(steps, get_losses()[i], color='black', alpha=0.2)
# plt.xlabel('Step', fontsize=8)
# plt.ylabel('Loss', fontsize=8)
# plt.title('Loss curves for cluster elements', fontsize=8)
# plt.xscale('log')
# plt.legend()

# st.pyplot(plt)

for i in get_clusters()[get_clusteri()]:
    idx = str(get_idxs()[i]) # the index into the pile tokens
    sample = get_contexts()[idx]
    context = sample['context']
    y = sample['y']
    tokens = context + [y]
    html = tokens_to_html(tokens)
    st.write("-----------------------------------------------------------")
    st.write(html, unsafe_allow_html=True)
