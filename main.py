# main.py
from fastapi import FastAPI, Query, APIRouter
from pydantic import BaseModel
import openai
import pandas as pd
from supabase import create_client
from neo4j import GraphDatabase
from fastapi.middleware.cors import CORSMiddleware
import os

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
@app.get("/causal-insights")
async def causal_insights():
    # Step 1: Retrieve causal edges with labels from Neo4j
    with driver.session() as session:
        result = session.run("""
            MATCH (a:CausalLearned)-[r:CAUSES]->(b:CausalLearned)
            RETURN a.name AS source, b.name AS target, labels(a) AS source_labels, labels(b) AS target_labels
        """)
        edges = []
        for record in result:
            edges.append({
                "source": record["source"],
                "target": record["target"],
                "source_label": record["source_labels"][-1],
                "target_label": record["target_labels"][-1]
            })

    # Step 2: Generate explanation for each edge using GPT
    table_with_reasoning = []
    for edge in edges:
        explanation_prompt = (
            f"In a citizen satisfaction dataset, we observe that {edge['source']} "
            f"(a {edge['source_label']}) causes {edge['target']} (a {edge['target_label']}). "
            f"Explain why this might be the case based on public service logic or real-world reasoning."
        )
        try:
            chat_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst skilled at interpreting causal graphs in public services."},
                    {"role": "user", "content": explanation_prompt}
                ],
                temperature=0.6
            )
            reasoning = chat_response.choices[0].message.content.strip()
        except Exception as e:
            reasoning = f"⚠️ Unable to generate explanation: {e}"

        table_with_reasoning.append({
            "source": edge["source"],
            "target": edge["target"],
            "reasoning": reasoning
        })

    # Step 3: Generate summary across the whole graph
    graph_summary = "\n".join([f"- {row['source']} causes {row['target']}" for row in table_with_reasoning])
    summary_prompt = f"""
    You are given the following causal relationships found in a citizen satisfaction dataset:

    {graph_summary}

    Please provide a summary of the three most important causal factors influencing satisfaction, and suggest practical improvements for policymakers.
    """
    try:
        summary_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a public service policy advisor analyzing causal data."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.5
        )
        summary_text = summary_response.choices[0].message.content.strip()
    except Exception as e:
        summary_text = f"⚠️ Summary generation failed: {e}"

    return {
        "summary": summary_text,
        "explanations": table_with_reasoning
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
