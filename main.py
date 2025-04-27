# main.py
from fastapi import FastAPI
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

# ✅ Optional Health Check Endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI server is live!"}
