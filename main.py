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

# ✅ Main Reasoning Endpoint
@app.post("/ask-ai")
async def ask_ai(req: QuestionRequest):
    user_question = req.question

    # Find causal path
    causal_edges = find_causal_path()
    causal_summary = " -> ".join([f"{src} -> {tgt}" for src, tgt in causal_edges]) if causal_edges else "No causal path found."

    # Create enhanced prompt
    full_prompt = f"""
    A user asked the following question about citizen satisfaction:

    '{user_question}'

    Based on the following learned causal paths:
    {causal_summary}

    Please generate a helpful, clear explanation.
    """

    # Generate AI Response
    ai_response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful public service reasoning expert specializing in citizen satisfaction analysis."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.5
    )

    answer_text = ai_response.choices[0].message.content

    return {
        "answer": answer_text,
        "causal_path": causal_edges
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
            "reasoning": match["reasoning"] if match else "No reasoning found."
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

    try:
        chat_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in public sector policy and causal reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        answer = chat_response.choices[0].message.content.strip()
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
    cypher_query = """
    CALL () {
        WITH $start_node AS officer_name
        MATCH (o:Entity {name: officer_name})<-[:HANDLED_BY]-(c:Case)
        RETURN c
        LIMIT 25
    }
    MATCH (c)-[r]->(e)
    RETURN DISTINCT c.name AS source, e.name AS target, type(r) AS type

    UNION

    CALL () {
        WITH $start_node AS officer_name
        MATCH (o:Entity {name: officer_name})<-[:HANDLED_BY]-(c:Case)
        RETURN c, o
        LIMIT 25
    }
    RETURN DISTINCT c.name AS source, o.name AS target, 'HANDLED_BY' AS type
    """

    try:
        with driver.session() as session:
            result = session.run(cypher_query, start_node=start_node)
            records = result.data()

        # Extract nodes and edges
        nodes_set = set()
        edges = []

        for record in records:
            nodes_set.add(record["source"])
            nodes_set.add(record["target"])
            edges.append({
                "source": record["source"],
                "target": record["target"],
                "type": record["type"]
            })

        nodes = list(nodes_set)
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
