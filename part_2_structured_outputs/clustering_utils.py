from sentence_transformers import SentenceTransformer
import umap
import numpy as np
import hdbscan
import sklearn.manifold
import numpy as np
import random
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CategoricalColorMapper
from bokeh.palettes import plasma, d3, Turbo256
from bokeh.plotting import figure
from bokeh.transform import transform
import bokeh.plotting as bpl
import instructor
from openai import OpenAI, AsyncOpenAI
from typing import List, Annotated
import weave
from pprint import pprint
from typing import Iterable
from enum import Enum
from typing_extensions import Literal
from tqdm.auto import tqdm
import asyncio 
from pydantic import ValidationInfo
from pydantic import BaseModel, AfterValidator, WithJsonSchema, Field

from clustering_prompts import clustering_prompt, cluster_cleanup_prompt, cluster_refine_prompt_create_true, cluster_refine_prompt_create_false

aclient = instructor.from_openai(AsyncOpenAI())
client = instructor.from_openai(OpenAI())


def cluster_texts(texts, model='sentence-transformers/all-MiniLM-L12-v2'):
    model = SentenceTransformer(model)
    embeddings = model.encode(texts)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=5, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    return cluster_labels, umap_embeddings

def visualize(texts, categories, embeddings):
    bpl.output_notebook()
    list_x = embeddings[:,0]
    list_y = embeddings[:,1]
    categories = [str(x) for x in categories] # required by color mapper
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

def cluster_examples(df, cid, n=5):
    descriptions = [x for x,y in zip(df.description, df.cluster_id) if y == cid]
    descriptions = random.choices(descriptions, k=n)
    out = []
    for x in descriptions:
        s = '<description>' + x + '</description>'
        out.append(s)
    return '\n'.join(out)

class Cluster(BaseModel):
    chain_of_thought: str = Field(..., description="Think step by step to come up with a cluster name")
    name: str

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

# for each cluster, extract 5 descriptions, ask LLM to name the cluster
# for all cluster names, ask LLM to make any corrections or add more names to make the list complete
# for each company / description and original cluster, given all clusters, choose a cluster or add a new cluster to the list

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

class UniqueClusters(BaseModel):
    chain_of_thought: str = Field(description="Think step by step before cleaning up the clusters")
    names: List[str] = Field(
        ...,
        description="List of deduplicated and cleaned up cluster names",
    )

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
                

class ClusterAssignment(BaseModel):
    chain_of_thought: str = Field(description="Think step by step then assign description to existing cluster or create a new one")
    cluster_name: str
    
def cluster_exists(v: str, info: ValidationInfo):
    context = info.context
    if context:
        context = context.get("existing_clusters")
        if v not in context:
            raise ValueError(f"Assigned cluster `{v}` not found in existing clusters provided by the user, only use provided clusters exactly.")
    return v

ClusterAssignmentValidated = Annotated[
    str,
    AfterValidator(cluster_exists),
    WithJsonSchema({
        "type": "string",
        "description": "Every cluster assignment needs to match exactly one of the cluster names provided by user."
    })
]

class ClusterAssignmentCreateFalse(BaseModel):
    chain_of_thought: str = Field(description="Think step by step then assign description to existing cluster")
    cluster_name: ClusterAssignmentValidated

@weave.op()
async def assign_cluster(id_, description, cluster_names, create=True):
    cluster_names_str = wrap_cluster_names(cluster_names)
    if create:
        cluster_refine_prompt_val = cluster_refine_prompt_create_true + cluster_names_str
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
    else:
        cluster_refine_prompt_val = cluster_refine_prompt_create_false + cluster_names_str
        cluster_assignment = await aclient.chat.completions.create(
            model="gpt-4-turbo",
            # response_model=ClusterAssignmentCreateFalse,
            response_model=ClusterAssignment,
            messages=[
                {
                    "role": "system",
                    "content": cluster_refine_prompt_val,
                },
                {"role": "user", "content": description},
            ],
            # validation_context={"existing_clusters": description},
            # max_retries=2,
        )

    assigned_cluster = cluster_assignment.cluster_name
    return {
        'id': id_,
        'description': description,
        'cluster_name': cluster_assignment.cluster_name, 
        'CoT': cluster_assignment.chain_of_thought,
    }

@weave.op()
async def assign_clusters(df, cluster_names, create=True):
    assigned_clusters = await asyncio.gather(
        *[assign_cluster(id_, description, cluster_names, create=create) for id_, description in zip(df.index.values, df.description.values)]
    )
    clustered_df = pd.DataFrame(data=assigned_clusters)
    df['new_cluster'] = clustered_df['cluster_name']
    df['CoT'] = clustered_df['CoT']
    return df

