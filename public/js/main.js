// DOM Elements
const queryInput = document.getElementById('queryInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = document.querySelector('.btn-text');
const loader = document.querySelector('.loader');

const logsSection = document.querySelector('.logs-section');
const logsBox = document.getElementById('logsBox');

const resultsSection = document.querySelector('.results-section');
const matchedNodeBox = document.getElementById('matchedNodeBox');
const nodeName = document.getElementById('nodeName');
const similarityScore = document.getElementById('similarityScore');

const contextBox = document.getElementById('contextBox');
const contextContent = document.getElementById('contextContent');

const answerBox = document.getElementById('answerBox');
const answerContent = document.getElementById('answerContent');

// Info Section Toggle
const toggleInfoBtn = document.getElementById('toggleInfoBtn');
const infoContent = document.getElementById('infoContent');

// PDF Viewer Toggle
const togglePdfBtn = document.getElementById('togglePdfBtn');
const pdfViewerContainer = document.getElementById('pdfViewerContainer');

// API Configuration - Use relative URLs (same origin)
const API_BASE_URL = '';  // Empty string = same origin

// State
let isProcessing = false;

// Event Listeners
analyzeBtn.addEventListener('click', handleAnalyze);

queryInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        handleAnalyze();
    }
});

// Info Section Toggle
if (toggleInfoBtn && infoContent) {
    toggleInfoBtn.addEventListener('click', () => {
        if (infoContent.style.display === 'none') {
            infoContent.style.display = 'block';
            toggleInfoBtn.classList.add('active');
        } else {
            infoContent.style.display = 'none';
            toggleInfoBtn.classList.remove('active');
        }
    });
}

// PDF Viewer Toggle
if (togglePdfBtn && pdfViewerContainer) {
    togglePdfBtn.addEventListener('click', () => {
        if (pdfViewerContainer.style.display === 'none') {
            pdfViewerContainer.style.display = 'block';
            togglePdfBtn.textContent = 'Hide Bharatiya Nyaya Sanhita Copy';
            // Scroll to PDF viewer
            pdfViewerContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            pdfViewerContainer.style.display = 'none';
            togglePdfBtn.textContent = 'Show Bharatiya Nyaya Sanhita Copy';
        }
    });
}

// Main Analysis Function
async function handleAnalyze() {
    const query = queryInput.value.trim();
    
    if (!query) {
        alert('Please enter a query or news article');
        return;
    }
    
    if (isProcessing) return;
    
    // Reset UI
    resetUI();
    
    // Show loading state
    setLoadingState(true);
    
    try {
        // Use streaming endpoint for better UX
        await analyzeWithStreaming(query);
        
    } catch (error) {
        addLog(`❌ Error: ${error.message}`, 'error');
        console.error('Analysis error:', error);
    } finally {
        setLoadingState(false);
    }
}

// Streaming Analysis
async function analyzeWithStreaming(query) {
    const response = await fetch(`${API_BASE_URL}/api/analyze-stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const data = JSON.parse(line.slice(6));
                    handleStreamEvent(data);
                } catch (e) {
                    console.error('Error parsing SSE data:', e);
                }
            }
        }
    }
}

// Handle Stream Events
function handleStreamEvent(data) {
    switch (data.type) {
        case 'log':
            addLog(data.message);
            break;
            
        case 'matched_node':
            showMatchedNode(data.node_name, data.similarity_score);
            break;
            
        case 'context':
            showContext(data.context);
            break;
            
        case 'answer':
            showAnswer(data.answer);
            break;
            
        case 'complete':
            addLog('✅ Analysis complete!', 'success');
            break;
            
        case 'error':
            addLog(`❌ Error: ${data.message}`, 'error');
            console.error(data.traceback);
            break;
    }
}

// Non-Streaming Analysis (Fallback)
async function analyzeWithoutStreaming(query) {
    addLog('Sending request...');
    
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query })
    });
    
    const result = await response.json();
    
    if (result.status === 'error') {
        throw new Error(result.error);
    }
    
    addLog('✅ Request successful');
    
    // Display results
    const { answer, matched_node, similarity_score, context } = result.data;
    
    showMatchedNode(matched_node, similarity_score);
    showContext(context);
    showAnswer(answer);
    
    addLog('✅ Analysis complete!', 'success');
}

// UI Helper Functions
function resetUI() {
    logsBox.innerHTML = '<p class="log-item">Starting analysis...</p>';
    logsSection.style.display = 'block';
    resultsSection.style.display = 'none';
    matchedNodeBox.style.display = 'none';
    contextBox.style.display = 'none';
    answerBox.style.display = 'none';
}

function setLoadingState(loading) {
    isProcessing = loading;
    analyzeBtn.disabled = loading;
    
    if (loading) {
        btnText.style.display = 'none';
        loader.style.display = 'block';
    } else {
        btnText.style.display = 'block';
        loader.style.display = 'none';
    }
}

function addLog(message, type = '') {
    const logItem = document.createElement('p');
    logItem.className = `log-item ${type}`;
    logItem.textContent = message;
    logsBox.appendChild(logItem);
    logsBox.scrollTop = logsBox.scrollHeight;
}

function showMatchedNode(name, score) {
    resultsSection.style.display = 'block';
    matchedNodeBox.style.display = 'block';
    nodeName.textContent = name;
    similarityScore.textContent = (score * 100).toFixed(2) + '%';
}

function showContext(context) {
    contextBox.style.display = 'block';
    contextContent.textContent = context;
}

function showAnswer(answer) {
    answerBox.style.display = 'block';
    
    // Convert markdown-like formatting to HTML
    let formattedAnswer = answer
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    
    answerContent.innerHTML = `<p>${formattedAnswer}</p>`;
    
    // Scroll to answer
    answerBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Health Check on Load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const health = await response.json();
        
        if (health.status === 'healthy') {
            console.log('✅ API is healthy', health);
        } else {
            console.warn('⚠️ API health check failed', health);
            alert('Warning: API may not be fully initialized. Some features may not work.');
        }
    } catch (error) {
        console.error('❌ Failed to connect to API', error);
        alert('Error: Cannot connect to backend API. Please ensure Flask is running on port 5000.');
    }
});
