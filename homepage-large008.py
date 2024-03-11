
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

def get_idxs():
    if 'idxs' not in st.session_state:
        with open("data-large008/idxs.pkl", "rb") as f:
            st.session_state['idxs'] = pickle.load(f)
        return st.session_state['idxs']
    else:
        return st.session_state['idxs']
    

def get_mean_loss():
    if 'mean_loss' not in st.session_state:
        st.session_state['mean_loss'] = np.load("data-large008/mean_loss_curve.npy")
        return st.session_state['mean_loss']
    else:
        return st.session_state['mean_loss']
    
def filter_clusters(search_query):
    if search_query:
        return [idx for idx in range(int(n_clusters)) if search_query.lower() in str(idx)]
    else:
        return range(int(n_clusters))

n_clusters = 4000

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

st.sidebar.write("These are clusters for pythia-70m-deduped, with a loss threshold of 0.3 nats, across 100k tokens, using k-means for clustering. I projected the gradients (~19m-dimensional) onto a 30k-dimensional subspace for clustering. There are 4000 clusters in total.")

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
with SqliteDict("data-large008/database.sqlite") as db:
    compressed_bytes = db[get_clusteri()]
    decompressed_object = io.BytesIO(compressed_bytes)
    with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
        cluster_data = pickle.load(file)

contexts = cluster_data['contexts']
losses = cluster_data['losses']
permuted_C = cluster_data['permuted_C']

# Add a download button for the contexts in the sidebar
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
    y = context['y']
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
im2 = ax2.imshow(permuted_C, cmap='rainbow', vmin=-1, vmax=1)
ax2.set_title('Similarity matrix for cluster', fontsize=10)
plt.colorbar(im2, ax=ax2)

# Subplot 3: Loss curves
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

for context in contexts.values():
    y = context['y']
    tokens = context['context'] + [y]
    html = tokens_to_html(tokens)
    st.write("-----------------------------------------------------------")
    st.write(html, unsafe_allow_html=True)
