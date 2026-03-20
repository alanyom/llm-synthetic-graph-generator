import os
import json
import random
import torch
import torch.nn.functional as F
from openai import OpenAI
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv

# Config
NUM_PEOPLE = 100
NUM_JOBS = 50
NUM_COMPANIES = 30
OUTPUT_FILE = "graph_dataset.json"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Dataset generation
prompt = f"""
Generate a JSON dataset for a heterogeneous graph with exactly {NUM_PEOPLE} person nodes,
{NUM_JOBS} job_posting nodes, and {NUM_COMPANIES} company nodes.

The JSON must have two top-level keys: "nodes" and "edges".

Each node must have:
- "id": a unique string (e.g. "P001" for persons, "J001" for jobs, "C001" for companies)
- "type": one of "person", "job_posting", or "company"

Person nodes must also include:
- "name": a realistic full name
- "gpa": a float between 2.0 and 4.0
- "skills": a list of 2-4 relevant technical skills
- "experience_years": an integer between 0 and 15

Job posting nodes must also include:
- "title": a job title
- "company_id": a valid company id from the dataset
- "required_skills": a list of 2-4 skills
- "seniority": one of "junior", "mid", "senior"

Company nodes must also include:
- "name": a realistic company name
- "industry": the industry sector
- "size": one of "startup", "mid-size", "enterprise"

Edges must include:
- "source": the id of the source node
- "target": the id of the target node
- "relation": one of "applied_to" (person -> job_posting), "worked_at" (person -> company),
  or "posted_by" (job_posting -> company)

Ensure all cross-references are valid (e.g. company_id in job nodes must exist in the node list).
Return only valid JSON with no additional commentary or markdown formatting.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)

dataset_text = response.choices[0].message.content.strip()

if dataset_text.startswith("```json"):
    dataset_text = dataset_text[len("```json"):]
if dataset_text.endswith("```"):
    dataset_text = dataset_text[:-3]

dataset = json.loads(dataset_text)

with open(OUTPUT_FILE, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Dataset saved to {OUTPUT_FILE}")

# Graph construction
data = HeteroData()

person_nodes = [n for n in dataset["nodes"] if n["type"] == "person"]
job_nodes = [n for n in dataset["nodes"] if n["type"] == "job_posting"]
company_nodes = [n for n in dataset["nodes"] if n["type"] == "company"]

person_ids = [p["id"] for p in person_nodes]
job_ids = [j["id"] for j in job_nodes]
company_ids = [c["id"] for c in company_nodes]

def map_id_to_index(ids):
    return {id_: idx for idx, id_ in enumerate(ids)}

person_map = map_id_to_index(person_ids)
job_map = map_id_to_index(job_ids)
company_map = map_id_to_index(company_ids)

# Node features
data["person"].x = torch.tensor(
    [[p.get("gpa", 0.0)] for p in person_nodes], dtype=torch.float
)
data["job_posting"].x = torch.rand((len(job_ids), 1))
data["company"].x = torch.rand((len(company_ids), 1))


edge_index = {
    "applied_to": [[], []],
    "worked_at": [[], []],
    "posted_by": [[], []]
}

for e in dataset["edges"]:
    src, tgt, rel = e["source"], e["target"], e["relation"]
    if rel == "applied_to" and src in person_map and tgt in job_map:
        edge_index["applied_to"][0].append(person_map[src])
        edge_index["applied_to"][1].append(job_map[tgt])
    elif rel == "worked_at" and src in person_map and tgt in company_map:
        edge_index["worked_at"][0].append(person_map[src])
        edge_index["worked_at"][1].append(company_map[tgt])
    elif rel == "posted_by" and src in job_map and tgt in company_map:
        edge_index["posted_by"][0].append(job_map[src])
        edge_index["posted_by"][1].append(company_map[tgt])

for key, (src_list, tgt_list) in edge_index.items():
    if src_list:
        data[key].edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)

# Heterogeneous GNN
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        self.conv1 = HeteroConv({
            ('person', 'applied_to', 'job_posting'): GCNConv(-1, hidden_channels),
            ('person', 'worked_at', 'company'): GCNConv(-1, hidden_channels),
            ('job_posting', 'posted_by', 'company'): GCNConv(-1, hidden_channels),
        }, aggr='sum')
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        person_emb = x_dict["person"]
        job_emb = x_dict["job_posting"]
        scores = torch.sigmoid(person_emb @ job_emb.T)
        return scores

# Training
model = HeteroGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

applied_src = edge_index["applied_to"][0]
applied_tgt = edge_index["applied_to"][1]

# Positive labels from actual applied_to edges
pos_labels = torch.ones(len(applied_src), dtype=torch.float)

# Negative samples: random person-job pairs not in applied_to edges
applied_set = set(zip(applied_src, applied_tgt))
neg_src, neg_tgt = [], []
while len(neg_src) < len(applied_src):
    p = random.randint(0, len(person_ids) - 1)
    j = random.randint(0, len(job_ids) - 1)
    if (p, j) not in applied_set:
        neg_src.append(p)
        neg_tgt.append(j)

neg_labels = torch.zeros(len(neg_src), dtype=torch.float)

all_src = applied_src + neg_src
all_tgt = applied_tgt + neg_tgt
all_labels = torch.cat([pos_labels, neg_labels])

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    scores = model(data.x_dict, {k: data[k].edge_index for k in data.edge_types})
    preds = scores[all_src, all_tgt]
    loss = F.binary_cross_entropy(preds, all_labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        predicted = (preds > 0.5).float()
        accuracy = (predicted == all_labels).float().mean().item()

    print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

print("\nGNN training complete.")
