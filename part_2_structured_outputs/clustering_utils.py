from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import pandas as pd
import random
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
from bokeh.transform import transform
import bokeh.plotting as bpl
import instructor
from openai import OpenAI, AsyncOpenAI
import weave
import asyncio 
from pydantic import BaseModel, Field
from typing import List

from clustering_prompts import clustering_prompt, cluster_cleanup_prompt, cluster_refine_prompt_create_true, cluster_refine_prompt_create_false

aclient = instructor.from_openai(AsyncOpenAI())
client = instructor.from_openai(OpenAI())
weave.init('llm-clustering')

# Function to cluster texts and return embeddings for visualization
def cluster_texts(texts, model='sentence-transformers/all-MiniLM-L12-v2'):
    model = SentenceTransformer(model)
    embeddings = model.encode(texts)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=5, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    return cluster_labels, umap_embeddings

# Function to visualize the clusters
def visualize(texts, categories, embeddings):
    bpl.output_notebook()
    list_x = embeddings[:,0]
    list_y = embeddings[:,1]
    categories = [str(x) for x in categories]
    clrs = random.sample(Turbo256, len(set(categories)), )
    color_map = CategoricalColorMapper(factors=list(set(categories)), palette=clrs)
    source = ColumnDataSource(data=dict(x=list_x, y=list_y, desc=texts, cat=categories))
    hover = HoverTool(tooltips=[
        ("index", "$index"),
        ("(x,y)", "(@x, @y)"),
        ('desc', '@desc'),
        ('cat', '@cat')
    ])
    hover.tooltips = """
    <div style="width:200px;">
    <div><strong>@cat</strong></div>
    <div>@desc</div>
    </div>
    """
    p = figure(tools=[hover], title="Embedding Visualization", width=800, height=600)
    p.min_border_left = 80
    p.circle('x', 'y', size=5, source=source, fill_color=transform('cat', color_map),)
    bpl.show(p)

# Function to select a subset of clusters and wrap them in xml format
def cluster_examples(df, cid, n=5):
    descriptions = [x for x,y in zip(df.description, df.cluster_id) if y == cid]
    descriptions = random.choices(descriptions, k=n)
    out = []
    for x in descriptions:
        s = '<description>' + x + '</description>'
        out.append(s)
    return '\n'.join(out)

# Pydantic model for cluster name
class Cluster(BaseModel):
    chain_of_thought: str = Field(description="Think step by step to come up with a cluster name")
    name: str

# Function to name a cluster based on examples
@weave.op()
async def name_cluster(df, cluster_id):
    cluster = await aclient.chat.completions.create(
        model="gpt-4-turbo",
        response_model=Cluster,
        messages=[
            {
                "role": "system",
                "content": clustering_prompt,
            },
            {"role": "user", "content": cluster_examples(df, cluster_id)},
        ],
    )
    return cluster

# Function to name all clusters in a dataframe
@weave.op()
async def name_clusters(df):
    cluster_ids = [x for x in df.cluster_id.unique() if x != -1]
    cluster_names = await asyncio.gather(
        *[name_cluster(df, cluster_id) for cluster_id in cluster_ids]
    )
    return [cluster.name for cluster in cluster_names]

# function to put all cluster names in xml format
def wrap_cluster_names(cnames):
    out = []
    for x in cnames:
        s = '<cluster>' + x + '</cluster>'
        out.append(s)
    return '\n'.join(out)

# Pydantic model for unique clusters
class UniqueClusters(BaseModel):
    chain_of_thought: str = Field(description="Think step by step before cleaning up the clusters")
    names: List[str] = Field(description="List of deduplicated and cleaned up cluster names")

# Function to deduplicate cluster names
@weave.op()
def _dedup_names(cluster_names):
    cluster_names_str = wrap_cluster_names(cluster_names)
    clean_clusters = client.chat.completions.create(
        model="gpt-4-turbo",
        response_model=UniqueClusters,
        messages=[
            {
                "role": "system",
                "content": cluster_cleanup_prompt,
            },
            {"role": "user", "content": cluster_names_str},
        ],
    )
    return clean_clusters.names

# Function to deduplicate cluster names that will split large inputs into sub-clusters
@weave.op()
def dedup_cluster_names(cluster_names):
    if len(cluster_names) < 100:
        return _dedup_names(cluster_names)
    else:
        new_names = []
        labels, _ = cluster_texts(cluster_names)
        for l in list(set(labels)):
            subset_names = [x for x,y in zip(cluster_names, labels) if y == l]
            clean_subset = _dedup_names(subset_names)
            new_names.extend(clean_subset)
        return _dedup_names(new_names)
                
# Pydantic model for cluster assignment
class ClusterAssignment(BaseModel):
    chain_of_thought: str = Field(description="Think step by step then assign description to existing cluster or create a new one")
    cluster_name: str
    
# Function to assign cluster to a description
@weave.op()
async def assign_cluster(id_, description, cluster_names, create=True):
    cluster_names_str = wrap_cluster_names(cluster_names)
    prompt = cluster_refine_prompt_create_true if create else cluster_refine_prompt_create_false
    cluster_refine_prompt_val = prompt + cluster_names_str
    cluster_assignment = await aclient.chat.completions.create(
        model="gpt-4-turbo",
        response_model=ClusterAssignment,
        messages=[
            {
                "role": "system",
                "content": cluster_refine_prompt_val,
            },
            {"role": "user", "content": description},
        ],
    )
    return {
        'id': id_,
        'description': description,
        'cluster_name': cluster_assignment.cluster_name, 
        'CoT': cluster_assignment.chain_of_thought,
    }

# Function to assign clusters to all descriptions in a dataframe
@weave.op()
async def assign_clusters(df, cluster_names, create=True):
    assigned_clusters = await asyncio.gather(
        *[assign_cluster(id_, description, cluster_names, create=create) for id_, description in zip(df.index.values, df.description.values)]
    )
    clustered_df = pd.DataFrame(data=assigned_clusters)
    df['new_cluster'] = clustered_df['cluster_name']
    df['CoT'] = clustered_df['CoT']
    return df

# Writes a list of cluster names to a specified file.
def write_cluster_names_to_file(filename, cluster_names):
    with open(filename, 'w') as file:
        file.write('\n'.join(cluster_names))