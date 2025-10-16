from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

class LegalInference:
    """Lightweight LLM inference for legal queries"""
    
    def __init__(self, model="llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.llm = ChatGroq(model=model, api_key=api_key, temperature=0.3)
        print(f"✅ LLM initialized: {model}")
        
        self.prompt = ChatPromptTemplate.from_template("""
You are a legal assistant AI specialized in the Bharatiya Nyaya Sanhita (BNS).

Given the retrieved legal context and matched offense, provide a structured legal interpretation.

Return the following:
- **Chapter Number and Name**
- **Section Number**
- **Punishment Clause(s)**
- **Brief Explanation**

Use ONLY the information provided in the context. Be precise and cite sections accurately.

---

**Matched Offense:** {offense}

**Retrieved Context:**
{context}

---

**Legal Interpretation:**
""")

    def generate_interpretation(self, context, offense_name):
        """Generate legal interpretation from context"""
        try:
            formatted_prompt = self.prompt.format_prompt(
                context=context,
                offense=offense_name
            ).to_messages()
            
            response = self.llm.invoke(formatted_prompt)
            return response.content
            
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return f"Error generating interpretation: {str(e)}"
