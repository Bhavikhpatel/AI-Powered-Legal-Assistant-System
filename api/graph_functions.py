from neo4j import GraphDatabase
import re
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import os

class Graphclass:
    def __init__(self, database="neo4j"):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = database
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.node_names = []
        self.node_embeddings = None

    def create_knowledge_graph(self, tuples):
        with self.driver.session(database=self.database) as session:
            for i, (offence, chapter, section, punishment) in enumerate(tuples):
                try:
                    session.execute_write(
                        self._create_offense_graph,
                        offence, chapter, section, punishment
                    )
                except Exception as e:
                    print(f"[Warning] Failed to create graph for tuple {i}: {offence, chapter, section, punishment}")
                    print("  Reason:", e)
        print("Graph creation complete.")

    @staticmethod
    def _create_offense_graph(tx, offence, chapter, section, punishment):
        query = """
MERGE (o:Offense {name: $offence})
MERGE (c:Chapter {name: 'Chapter No.: ' + $chapter})
MERGE (s:Section {number: 'Section No.: ' + $section})
MERGE (p:Punishment {description: $punishment})

MERGE (o)-[:refersToChapter]->(c)
MERGE (o)-[:refersToSection]->(s)
MERGE (o)-[:hasPunishment]->(p)
"""
        tx.run(query, offence=offence, chapter=chapter, section=section, punishment=punishment)

    def _fetch_all_node_names(self):
        def fetch_all_node_names(tx):
            query = """
            MATCH (n)
            WHERE n.name IS NOT NULL
            RETURN DISTINCT n.name AS name
            """
            result = tx.run(query)
            return [record["name"] for record in result]

        with self.driver.session(database=self.database) as session:
            node_names = session.execute_read(fetch_all_node_names)

        self.node_names = [str(name) for name in node_names]

    def _encode_node_names(self):
        if not self.node_names:
            self._fetch_all_node_names()
        self.node_embeddings = self.model.encode(self.node_names, convert_to_numpy=True)

    def find_most_similar_node(self, input_text):
        if self.node_embeddings is None:
            self._encode_node_names()

        input_embedding = self.model.encode([input_text], convert_to_numpy=True)[0]

        similarities = [1 - cosine(input_embedding, node_emb) for node_emb in self.node_embeddings]

        best_idx = np.argmax(similarities)

        return self.node_names[best_idx], similarities[best_idx]

    def fetch_related_info(self, node_name):
        def fetch_related_info_tx(tx, node_name):
            query = """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($node_name)
              OR toLower(n.number) CONTAINS toLower($node_name)
            WITH n
            MATCH (n)-[*1..2]-(related)
            RETURN DISTINCT coalesce(related.name, related.number, related.description, '') AS info,
                            labels(related) AS labels
            """
            result = tx.run(query, node_name=node_name)
            return [{"info": record["info"], "labels": record["labels"]} for record in result]

        with self.driver.session() as session:
            return session.read_transaction(fetch_related_info_tx, node_name)

    def get_context_text_for_llm(self, node_name):
        related_infos = self.fetch_related_info(node_name)

        if not related_infos:
            print(f"[DEBUG] No related info found for node: {node_name}")
            return f"No context found for: {node_name}"

        context_texts = [item["info"] for item in related_infos if item["info"] and item["info"].strip() != ""]

        if not context_texts:
            print(f"[DEBUG] Related nodes found but no valid 'info' properties for: {node_name}")
            return f"No usable info for: {node_name}"

        context = "\n".join(context_texts)
        return context
    