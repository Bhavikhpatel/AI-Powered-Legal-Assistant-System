from neo4j import GraphDatabase
import os
import requests
import numpy as np
from scipy.spatial.distance import cosine

class GraphQuery:
    """Lightweight class for querying existing Neo4j knowledge graph"""
    
    def __init__(self, database="neo4j"):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = database
        
        if not self.uri or not self.user or not self.password:
            raise ValueError("Neo4j credentials not found")
        
        print(f"🔌 Connecting to Neo4j...")
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                max_connection_lifetime=3600
            )
            
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (n:Offense) RETURN count(n) as count")
                count = result.single()["count"]
                print(f"✅ Connected! Found {count} offenses in knowledge graph")
            
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Cannot connect to Neo4j: {str(e)}")

        # Hugging Face API for embeddings
        self.hf_api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.hf_headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        print("✅ Query system initialized")

        self.node_names = []
        self.node_embeddings = None

    def _get_embeddings(self, texts):
        """Get embeddings from Hugging Face API"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = requests.post(
                self.hf_api_url,
                headers=self.hf_headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=30
            )
            
            if response.status_code == 200:
                return np.array(response.json())
            else:
                print(f"⚠️ HF API error: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
        
        # Fallback: random embeddings
        return np.random.rand(len(texts), 384)

    def fetch_all_offenses(self):
        """Fetch all offense names from the graph"""
        def fetch_tx(tx):
            query = "MATCH (n:Offense) RETURN DISTINCT n.name AS name"
            result = tx.run(query)
            return [record["name"] for record in result]

        with self.driver.session(database=self.database) as session:
            self.node_names = session.execute_read(fetch_tx)
        
        print(f"📊 Loaded {len(self.node_names)} offenses")
        return self.node_names

    def encode_offenses(self):
        """Encode all offenses to embeddings"""
        if not self.node_names:
            self.fetch_all_offenses()
        
        if not self.node_names:
            raise ValueError("No offenses found in knowledge graph")
        
        print("🔢 Encoding offenses...")
        self.node_embeddings = self._get_embeddings(self.node_names)
        print(f"✅ Encoded {len(self.node_embeddings)} embeddings")

    def find_most_similar_offense(self, query_text):
        """Find the most relevant offense for a query"""
        if self.node_embeddings is None:
            self.encode_offenses()

        query_embedding = self._get_embeddings(query_text)[0]

        similarities = [
            1 - cosine(query_embedding, offense_emb) 
            for offense_emb in self.node_embeddings
        ]

        best_idx = np.argmax(similarities)
        return self.node_names[best_idx], float(similarities[best_idx])

    def get_offense_context(self, offense_name):
        """Get full legal context for an offense"""
        def fetch_context_tx(tx, offense_name):
            query = """
            MATCH (o:Offense {name: $offense_name})
            OPTIONAL MATCH (o)-[:refersToChapter]->(c:Chapter)
            OPTIONAL MATCH (o)-[:refersToSection]->(s:Section)
            OPTIONAL MATCH (o)-[:hasPunishment]->(p:Punishment)
            RETURN 
                o.name as offense,
                c.name as chapter,
                s.number as section,
                p.description as punishment
            """
            result = tx.run(query, offense_name=offense_name)
            return result.single()

        with self.driver.session(database=self.database) as session:
            record = session.execute_read(fetch_context_tx, offense_name)
            
            if not record:
                return f"No context found for: {offense_name}"
            
            context_parts = []
            if record["chapter"]:
                context_parts.append(record["chapter"])
            if record["section"]:
                context_parts.append(record["section"])
            if record["punishment"]:
                context_parts.append(f"Punishment: {record['punishment']}")
            
            return "\n".join(context_parts) if context_parts else "No context available"
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j connection closed")
