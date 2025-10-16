from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import os
from dotenv import load_dotenv
import traceback
import json

load_dotenv()

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

CORS(app)

# Import query modules
from api.graph_query import GraphQuery
from api.inference import LegalInference
from api.utils import split_think_sections

# Global instances
graph_query = None
legal_llm = None

def initialize_system():
    """Initialize query system and LLM"""
    global graph_query, legal_llm
    try:
        print("🚀 Initializing BNS Legal Assistant...")
        graph_query = GraphQuery()
        legal_llm = LegalInference()
        print("✅ System ready!")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        traceback.print_exc()
        return False

# Initialize on startup
initialize_system()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "graph_connected": graph_query is not None,
        "llm_ready": legal_llm is not None
    }), 200


@app.route('/api/analyze', methods=['POST'])
def analyze_query():
    """Analyze legal query (non-streaming)"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "error": "Missing 'query' in request body"
            }), 400
        
        query_text = data['query']
        
        if not query_text.strip():
            return jsonify({
                "status": "error",
                "error": "Query cannot be empty"
            }), 400
        
        if graph_query is None or legal_llm is None:
            return jsonify({
                "status": "error",
                "error": "System not initialized"
            }), 500
        
        # Find matching offense
        offense_name, similarity_score = graph_query.find_most_similar_offense(query_text)
        
        # Get legal context
        context = graph_query.get_offense_context(offense_name)
        
        # Generate interpretation
        answer = legal_llm.generate_interpretation(context, offense_name)
        _, clean_answer = split_think_sections(answer)
        
        return jsonify({
            "status": "success",
            "data": {
                "answer": clean_answer,
                "matched_node": offense_name,
                "similarity_score": similarity_score,
                "context": context
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/analyze-stream', methods=['POST'])
def analyze_query_stream():
    """Streaming analysis with real-time updates"""
    try:
        data = request.get_json()
        query_text = data.get('query', '') if data else ''
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Failed to parse request: {str(e)}"
        }), 400
    
    if not query_text.strip():
        return jsonify({
            "status": "error",
            "error": "Query cannot be empty"
        }), 400
    
    if graph_query is None or legal_llm is None:
        return jsonify({
            "status": "error",
            "error": "System not initialized"
        }), 500
    
    def generate(query):
        try:
            yield f"data: {json.dumps({'type': 'log', 'message': 'Searching knowledge graph...'})}\n\n"
            
            offense_name, similarity_score = graph_query.find_most_similar_offense(query)
            
            yield f"data: {json.dumps({'type': 'log', 'message': f'Found: {offense_name} ({similarity_score:.2%} match)'})}\n\n"
            yield f"data: {json.dumps({'type': 'matched_node', 'node_name': offense_name, 'similarity_score': similarity_score})}\n\n"
            
            yield f"data: {json.dumps({'type': 'log', 'message': 'Retrieving legal context...'})}\n\n"
            
            context = graph_query.get_offense_context(offense_name)
            
            yield f"data: {json.dumps({'type': 'context', 'context': context})}\n\n"
            
            yield f"data: {json.dumps({'type': 'log', 'message': 'Generating interpretation...'})}\n\n"
            
            answer = legal_llm.generate_interpretation(context, offense_name)
            _, clean_answer = split_think_sections(answer)
            
            yield f"data: {json.dumps({'type': 'answer', 'answer': clean_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"
    
    response = Response(generate(query_text), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
