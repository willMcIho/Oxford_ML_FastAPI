# main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import openai
import pandas as pd
from supabase import create_client
from neo4j import GraphDatabase
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List

# ✅ Load from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ✅ Initialize clients
openai_client = openai.OpenAI(api_key=OPENAI_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS (Allow frontend like Lovable to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define input model
class QuestionRequest(BaseModel):
    question: str

# ✅ Helper to get simple causal path
def find_causal_path():
    with driver.session() as session:
        result = session.run("""
            MATCH p=(start:CausalLearned)-[:CAUSES*1..2]->(end:CausalLearned)
            RETURN start.name AS source, end.name AS target
            LIMIT 5
        """)
        return [(record["source"], record["target"]) for record in result]

ENTITY_KEYWORDS = {
    "officer": "Officer",
    "region":  "Region",
    "benefit": "BenefitType",
    "case":    "Case",
    "month":   "Month",
    "disability": "DisabilityStatus",
    "satisfaction": "Satisfaction"
}

def detect_entity_types(question: str):
    q = question.lower()
    found = set()
    for kw, label in ENTITY_KEYWORDS.items():
        if kw in q:
            found.add(label)
    return list(found)

def fetch_context(label: str):
    if label == "Officer":
        cypher = """
          MATCH (o:Officer)<-[:HANDLED_BY]-(c:Case)
          RETURN o.name AS entity, count(c) AS volume
          ORDER BY volume DESC
          LIMIT 3
        """
    elif label == "Region":
        cypher = """
          MATCH (r:Region)<-[:OCCURS_IN]-(c:Case)
          RETURN r.name AS entity, count(c) AS volume
          ORDER BY volume DESC
          LIMIT 3
        """
    elif label == "BenefitType":
        cypher = """
          MATCH (b:BenefitType)<-[:APPLIES_TO]-(c:Case)
          RETURN b.name AS entity, count(c) AS volume
          ORDER BY volume DESC
          LIMIT 3
        """
    # …and so on for Month, DisabilityStatus, Case (recent cases?), Satisfaction (distribution)
    else:
        return []

    rows = session.run(cypher).data()
    return rows


# ✅ Main Reasoning Endpoint
@app.post("/ask-ai")
async def ask_ai(req: QuestionRequest):
    q = req.question

    # 1. Detect entity types
    types = detect_entity_types(q)  # e.g. ["Officer","Region"]

    # 2. Fetch context snippets
    context_snippets = []
    for t in types:
        rows = fetch_context(t)
        if rows:
            snippet = f"Top {t}s by case volume:\n" + \
                      "\n".join([f"- {r['entity']}: {r['volume']} cases" for r in rows])
            context_snippets.append(snippet)

    # 3. Causal path summary
    causal_edges = find_causal_path()
    causal_summary = "\n".join([f"- {s} → {t}" for s, t in causal_edges])

    # 4. Pull case data
    response = supabase.table("Cases") \
        .select("ResolutionTime, Satisfaction, RequestDate, Region, Officer, Disability") \
        .execute()
    df = pd.DataFrame(response.data)

    # Ensure RequestDate is in datetime format
    df["RequestDate"] = pd.to_datetime(df["RequestDate"], errors="coerce")

    # Derive RequestMonth as full month name (e.g., "January")
    df["RequestMonth"] = df["RequestDate"].dt.strftime('%B')

    # 5. Compute enhanced metrics
    avg_res = df["ResolutionTime"].mean()
    med_res = df["ResolutionTime"].median()
    p90_res = df["ResolutionTime"].quantile(0.9)
    sat_dist = df["Satisfaction"].value_counts(normalize=True).round(2).to_dict()

    monthly = df.groupby("RequestMonth")["ResolutionTime"].mean().round(1).to_dict()
    region_times = df.groupby("Region")["ResolutionTime"].mean().round(1).to_dict()
    disability_sat = (
        df.groupby("Disability")["Satisfaction"]
        .value_counts(normalize=True).unstack().round(2).to_dict()
    )
    region_sat = (
        df.groupby("Region")["Satisfaction"]
        .value_counts(normalize=True).unstack().round(2).to_dict()
    )
    officer_means = df.groupby("Officer")["ResolutionTime"].mean()
    top_q, bot_q = officer_means.quantile([0.75, 0.25])
    equity_gap = round(top_q - bot_q, 1)
    disability_res = df.groupby("Disability")["ResolutionTime"].mean().round(1).to_dict()

    # 6. Build the metrics text block
    metrics_block = f"""
=== Overall Metrics ===
• Avg. resolution: {avg_res:.1f} days  
• Median resolution: {med_res:.1f} days  
• 90th pctile resolution: {p90_res:.1f} days  
• Satisfaction distribution: {', '.join(f'{k}: {v*100:.0f}%' for k,v in sat_dist.items())}

=== Monthly Resolution Trends ===
{"; ".join(f'{m}: {d:.1f}d' for m,d in monthly.items())}

=== Officer Equity ===
• 75th pctile officer res time: {top_q:.1f}d  
• 25th pctile officer res time: {bot_q:.1f}d  
• Equity gap: {equity_gap:.1f} days

=== Avg ResolutionTime by Region ===
{"; ".join(f'{k}: {v}d' for k,v in region_times.items())}

=== Satisfaction by Disability ===
{"; ".join(f'{k}: ' + ', '.join(f'{s}={v*100:.0f}%' for s,v in sv.items()) for k,sv in disability_sat.items())}

=== Satisfaction by Region ===
{"; ".join(f'{k}: ' + ', '.join(f'{s}={v*100:.0f}%' for s,v in sv.items()) for k,sv in region_sat.items())}

=== Avg ResolutionTime by Disability ===
{"; ".join(f'{k}: {v}d' for k,v in disability_res.items())}
"""

    # 7. Build the final prompt
    prompt = f"""
A user asked:
“{q}”

=== Learned Causal Paths ===
{causal_summary}

=== Dataset Metrics ===
{metrics_block}

"""

    if context_snippets:
        prompt += "\n".join([f"=== {s} ===\n{snippet}\n"
                             for s, snippet in zip(types, context_snippets)])

    prompt += "\nPlease provide a concise, actionable answer grounded in the above.\n"

    # 8. Create message structure
    messages=[
      # System role clarifies persona, style guidelines, and domain context  
      {
        "role": "system",
        "content": (
          "You are an expert public-service analyst specialising in UK Department "
          "for Work and Pensions (DWP) operations, benefit policy, and process improvement. "
          "• Use evidence-based reasoning.\n"
          "• Cite data from the prompt (aggregates, causal edges) explicitly.\n"
          "• Provide numbered recommendations.\n"
          "• Where relevant, reference specific benefit lines (e.g. PIP, UC).\n"
          "• Output in **Markdown** with sections: Summary ▶ Analysis ▶ Recommendations."
        )
      },
    
      # Optional “assistant” priming: shows the answer style you expect
      {
        "role": "assistant",
        "content": (
          "Understood. I will provide structured, evidence-backed insights with "
          "clear UK context. Ready for the user question."
        )
      },
    
      # User question and your assembled prompt block  
      {"role": "user", "content": prompt},
    ]

    # 8. Generate answer
    openai_resp = openai_client.chat.completions.create(
        model="gpt-4o",                # or gpt-4o-mini / gpt-4-0125-preview
        temperature=0.45,
        top_p=0.9,
        presence_penalty=0.2,
        messages=messages
    )

    answer = openai_resp.choices[0].message.content.strip()

    return {
        "question": q,
        "entity_context": context_snippets,
        "causal_paths": causal_edges,
        "dataset_metrics": {
            "avg_resolution": avg_res,
            "median_resolution": med_res,
            "p90_resolution": p90_res,
            "satisfaction_distribution": sat_dist,
            "monthly_avg_resolution": monthly,
            "region_avg_resolution": region_times,
            "region_satisfaction": region_sat,
            "disability_avg_resolution": disability_res,
            "disability_satisfaction": disability_sat,
            "officer_equity_gap": equity_gap
        },
        "answer": answer
    }

# Fetch causal graph edges
@app.get("/causal-graph")
async def causal_graph():
    with driver.session() as session:
        result = session.run("""
            MATCH (a:CausalLearned)-[r:CAUSES]->(b:CausalLearned)
            RETURN a.name AS source,
                   b.name AS target,
                   labels(a) AS source_labels,
                   labels(b) AS target_labels,
                   type(r) AS predicate
        """)
        edges = []
        for record in result:
            edges.append({
                "source": record["source"],
                "target": record["target"],
                "predicate": record["predicate"],
                "source_labels": record["source_labels"],
                "target_labels": record["target_labels"]
            })
    return {"edges": edges}

# AI reasoning over causal graph
@app.post("/causal-insights")
async def causal_insights(req: QuestionRequest):

    user_question = req.question

    # Step 1: Fetch causal edges from Neo4j
    with driver.session() as session:
        result = session.run("""
            MATCH (a:CausalLearned)-[r:CAUSES]->(b:CausalLearned)
            RETURN a.name AS source, b.name AS target, labels(a) AS source_labels, labels(b) AS target_labels
        """)
        neo4j_edges = [
            {
                "source": record["source"],
                "target": record["target"],
                "source_label": record["source_labels"][-1],
                "target_label": record["target_labels"][-1]
            }
            for record in result
        ]

    # Step 2: Fetch reasoning data from Supabase
    supabase_data = supabase.table("EdgeGraphsReasoning").select("*").execute()
    if not supabase_data.data:
        supabase_reasoning = []
    else:
        supabase_reasoning = supabase_data.data

    # Step 3: Merge Neo4j edges with Supabase reasoning
    explanations = []
    for edge in neo4j_edges:
        match = next(
            (r for r in supabase_reasoning if r.get("Source") == edge["source"] and r.get("Target") == edge["target"]),
            None
        )
        explanations.append({
            "source": edge["source"],
            "target": edge["target"],
            "source_label": edge["source_label"],
            "target_label": edge["target_label"],
            "reasoning": match.get("Reasoning Summary", "No reasoning found.") if match else "No reasoning found."
        })

    # Step 4: Prepare prompt
    formatted_edges = "\n".join([f"- {e['source']} → {e['target']} ({e['source_label']} → {e['target_label']})" for e in explanations])
    reasoning_blocks = "\n\n".join([f"{e['source']} → {e['target']}: {e['reasoning']}" for e in explanations])

    prompt = f"""
You are a public policy analyst. Here are learned causal relationships in a citizen service dataset:

Causal Edges:
{formatted_edges}

Reasoning for these causal relationships:
{reasoning_blocks}

User Question:
{user_question}

Please respond with a thoughtful, policy-relevant answer based on the causal graph and explanations above.
"""

# 5. Create message structure
    messages=[
      # System role clarifies persona, style guidelines, and domain context  
      {
        "role": "system",
        "content": (
          "You are an expert public-service analyst specialising in UK Department "
          "for Work and Pensions (DWP) operations, benefit policy, and process improvement. "
          "• Use evidence-based reasoning.\n"
          "• Cite data from the prompt (aggregates, causal edges) explicitly.\n"
          "• Provide numbered recommendations.\n"
          "• Where relevant, reference specific benefit lines (e.g. PIP, UC).\n"
          "• Output in **Markdown** with sections: Summary ▶ Analysis ▶ Recommendations."
        )
      },
    
      # Optional “assistant” priming: shows the answer style you expect
      {
        "role": "assistant",
        "content": (
          "Understood. I will provide structured, evidence-backed insights with "
          "clear UK context. Ready for the user question."
        )
      },
    
      # User question and your assembled prompt block  
      {"role": "user", "content": prompt},
    ]

    # 8. Generate answer
    openai_resp = openai_client.chat.completions.create(
        model="gpt-4o",                # or gpt-4o-mini / gpt-4-0125-preview
        temperature=0.45,
        top_p=0.9,
        presence_penalty=0.2,
        messages=messages
    )

    answer = openai_resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"⚠️ GPT response failed: {e}"

    return {
        "question": user_question,
        "response": answer,
        "explanations": explanations
    }

# Model for graph data
class GraphResponse(BaseModel):
    nodes: list
    edges: list

@app.get("/knowledge-graph")
async def get_knowledge_graph(start_node: str = Query(..., description="Entity to start from")):
    try:
        with driver.session() as session:
            # First, get the label of the node (assumes only 1 label is relevant)
            label_result = session.run("""
                MATCH (n {name: $start_node})
                RETURN head(labels(n)) AS label
            """, {"start_node": start_node}).single()

            if not label_result:
                return {"nodes": [], "edges": []}

            node_label = label_result["label"]

            if node_label == "Case":
                # If it's a Case, get outbound relationships
                query = """
                    MATCH (start:Case {name: $start_node})
                    CALL {
                        WITH start
                        MATCH (start)-[r2]->(e)
                        RETURN r2, e
                        LIMIT 50
                    }
                    WITH COLLECT(DISTINCT start) + COLLECT(DISTINCT e) AS all_nodes,
                         COLLECT(DISTINCT {source: start.name, target: e.name, type: type(r2)}) AS all_edges
                    UNWIND all_nodes AS n
                    RETURN DISTINCT n.name AS id, labels(n)[0] AS label, head(labels(n)) AS group, all_edges
                """
            else:
                # If it's an entity like Officer/Benefit/etc., get inbound Case relationships
                query = """
                    MATCH (start {name: $start_node})
                    CALL {
                        WITH start
                        MATCH (start)<-[r1]-(c:Case)
                        RETURN r1, c
                        LIMIT 50
                    }
                    OPTIONAL MATCH (c)-[r2]->(e)
                    WITH COLLECT(DISTINCT start) + COLLECT(DISTINCT c) + COLLECT(DISTINCT e) AS all_nodes,
                         COLLECT(DISTINCT {source: start.name, target: c.name, type: type(r1)}) +
                         COLLECT(DISTINCT {source: c.name, target: e.name, type: type(r2)}) AS all_edges
                    UNWIND all_nodes AS n
                    RETURN DISTINCT n.name AS id, labels(n)[0] AS label, head(labels(n)) AS group, all_edges
                """

            result = session.run(query, {"start_node": start_node})
            records = result.data()

        # Build nodes and edges
        nodes = []
        seen_nodes = set()
        edges = []
        seen_edges = set()

        for record in records:
            node_id = record["id"]
            if node_id and node_id not in seen_nodes:
                seen_nodes.add(node_id)
                nodes.append({
                    "id": node_id,
                    "label": node_id,
                    "group": record["group"]
                })

            for edge in record["all_edges"]:
                if edge and edge["source"] and edge["target"]:
                    edge_key = (edge["source"], edge["target"], edge["type"])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(edge)

        return {"nodes": nodes, "edges": edges}

    except Exception as e:
        return {"error": str(e)}


# Expand a node
@app.get("/knowledge-graph/expand", response_model=GraphResponse)
async def expand_node(node: str = Query(..., description="Node to expand")):
    with driver.session() as session:
        result = session.run("""
            MATCH (a {name: $node})-[r]->(b)
            RETURN a.name AS source, type(r) AS relationship, b.name AS target, labels(b) AS labels
            LIMIT 20
        """, {"node": node})

        nodes = set()
        edges = []
        for record in result:
            nodes.add(record["source"])
            nodes.add(record["target"])
            edges.append({
                "source": record["source"],
                "target": record["target"],
                "type": record["relationship"]
            })

    return {"nodes": list(nodes), "edges": edges}

# ✅ Optional Health Check Endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI server is live!"}
