from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import os
from dotenv import load_dotenv
import traceback
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

load_dotenv()

# Get the parent directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

CORS(app)

# Global variables for components
graph = None
inference_engine = None

def initialize_components():
    """Initialize graph and inference engine"""
    global graph, inference_engine
    try:
        from api.graph_functions import Graphclass
        from api.inference import Inference
        
        graph = Graphclass()
        inference_engine = Inference(api_key=os.getenv("GROQ_API_KEY"))
        print("✅ Components initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "graph_initialized": graph is not None,
        "inference_initialized": inference_engine is not None
    }), 200


@app.route('/api/analyze', methods=['POST'])
def analyze_query():
    """Analyze legal query endpoint (non-streaming)"""
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
        
        # Check if components are initialized
        if graph is None or inference_engine is None:
            return jsonify({
                "status": "error",
                "error": "Backend components not initialized. Please restart the server."
            }), 500
        
        # Find most similar node
        similar_node = graph.find_most_similar_node(query_text)
        node_name = similar_node[0]
        similarity_score = float(similar_node[1])
        
        # Get context for LLM
        context = graph.get_context_text_for_llm(node_name)
        
        # Generate answer
        answer = inference_engine.answer_user_question(context, node_name)
        
        # Remove think tags if present
        from api.utils import split_think_sections
        _, clean_answer = split_think_sections(answer)
        
        return jsonify({
            "status": "success",
            "data": {
                "answer": clean_answer,
                "matched_node": node_name,
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
    """Streaming version of analyze endpoint (Server-Sent Events)"""
    
    # CRITICAL: Get request data BEFORE the generator function
    try:
        data = request.get_json()
        query_text = data.get('query', '') if data else ''
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Failed to parse request: {str(e)}"
        }), 400
    
    # Validate query
    if not query_text.strip():
        return jsonify({
            "status": "error",
            "error": "Query cannot be empty"
        }), 400
    
    # Check if components are initialized
    if graph is None or inference_engine is None:
        return jsonify({
            "status": "error",
            "error": "Backend components not initialized. Please restart the server."
        }), 500
    
    # Generator function receives query as parameter
    def generate(query):
        try:
            # Step 1: Finding similar node
            yield f"data: {json.dumps({'type': 'log', 'message': 'Finding most similar node...'})}\n\n"
            
            similar_node = graph.find_most_similar_node(query)
            node_name = similar_node[0]
            similarity_score = float(similar_node[1])
            
            yield f"data: {json.dumps({'type': 'log', 'message': f'Similar Node Found: {node_name} (Score: {similarity_score:.2f})'})}\n\n"
            yield f"data: {json.dumps({'type': 'matched_node', 'node_name': node_name, 'similarity_score': similarity_score})}\n\n"
            
            # Step 2: Fetching context
            yield f"data: {json.dumps({'type': 'log', 'message': 'Fetching context for LLM...'})}\n\n"
            
            context = graph.get_context_text_for_llm(node_name)
            
            yield f"data: {json.dumps({'type': 'log', 'message': 'Context fetched successfully.'})}\n\n"
            yield f"data: {json.dumps({'type': 'context', 'context': context})}\n\n"
            
            # Step 3: Generating answer
            yield f"data: {json.dumps({'type': 'log', 'message': 'Running inference...'})}\n\n"
            
            answer = inference_engine.answer_user_question(context, node_name)
            
            from api.utils import split_think_sections
            _, clean_answer = split_think_sections(answer)
            
            yield f"data: {json.dumps({'type': 'log', 'message': 'Inference completed.'})}\n\n"
            yield f"data: {json.dumps({'type': 'answer', 'answer': clean_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'traceback': error_trace})}\n\n"
    
    # Return streaming response with headers
    response = Response(generate(query_text), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print(f"📂 Template folder: {app.template_folder}")
    print(f"📂 Static folder: {app.static_folder}")
    print("🌐 Access the app at: http://127.0.0.1:5000")
    
    # Initialize components before starting the server
    # initialize_components()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
