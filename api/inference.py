import ast
import re
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate  # ← Fixed import
from dotenv import load_dotenv
import os

load_dotenv()

class Inference:
    def __init__(self, model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), batch_size=10):
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.batch_size = batch_size
        self.prompt = ChatPromptTemplate.from_template("""
Extract legal information tuples from the following text.
Return ONLY a valid Python list of tuples in this format:

(offence, chapter, section, punishment_clause)

Here are some detailed examples:

Text:
\"\"\"
1. Offence: Theft; Chapter: 5; Section: 378; Punishment: Imprisonment up to 3 years or fine, or both.
2. Offence: Criminal Breach of Trust; Chapter: 7; Section: 405; Punishment: Imprisonment for up to 2 years, or fine, or both.
\"\"\"
Output:
[
    ("Theft", "5", "378", "Imprisonment up to 3 years or fine, or both"),
    ("Criminal Breach of Trust", "7", "405", "Imprisonment for up to 2 years, or fine, or both")
]

Text:
\"\"\"
The offence of Murder falls under Chapter 6, Section 302. The punishment prescribed is death penalty or life imprisonment.
Attempt to murder is dealt with in Chapter 6, Section 307, punishable by imprisonment for up to 10 years and fine.
\"\"\"
Output:
[
    ("Murder", "6", "302", "Death penalty or life imprisonment"),
    ("Attempt to murder", "6", "307", "Imprisonment for up to 10 years and fine")
]

Text:
\"\"\"
Offence: Cheating; Chapter: 8; Section: 420; Punishment: Imprisonment which may extend to seven years and fine.
\"\"\"
Output:
[
    ("Cheating", "8", "420", "Imprisonment which may extend to seven years and fine")
]

Now extract from the following text:

\"\"\"{input_text}\"\"\"
""")

    def _safe_llm_invoke(self, prompt):
        for _ in range(3):
            try:
                return self.llm.invoke(prompt)
            except Exception:
                time.sleep(1)
        return None

    def extract_custom_tuples(self, chunks):
        all_tuples = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[batch_idx:batch_idx + self.batch_size]

            combined_text = "\n\n".join(
                [f"[CHUNK {i + batch_idx + 1}]\n{chunk.page_content}" for i, chunk in enumerate(batch_chunks)]
            )

            formatted_prompt = self.prompt.format_prompt(input_text=combined_text).to_messages()
            response = self._safe_llm_invoke(formatted_prompt)
            if not response:
                continue

            output_str = response.content

            matches = re.findall(r"\[\s*\([^)]+\)\s*(?:,\s*\([^)]+\)\s*)*\]", output_str, re.DOTALL)
            batch_tuples = []

            for match in matches:
                try:
                    tuples = ast.literal_eval(match)
                    if isinstance(tuples, list) and all(len(t) == 4 for t in tuples):
                        batch_tuples.extend(tuples)
                except:
                    pass

            all_tuples.extend(batch_tuples)
            print(f"Batch {(batch_idx // self.batch_size) + 1} finished")

        unique_tuples = list(set(all_tuples))
        return unique_tuples

    def save_tuples_to_file(self, tuples, filename="graphData.txt"):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("[\n")
            for t in tuples:
                f.write(f"    {repr(t)},\n")
            f.write("]\n")

    def load_tuples_from_file(self, filename="graphData.txt"):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                tuples = ast.literal_eval(content)
                if isinstance(tuples, list) and all(isinstance(t, tuple) for t in tuples):
                    return tuples
                else:
                    print("[Warning] File content is not a list of tuples.")
                    return []
            except Exception as e:
                print(f"[Error] Failed to parse file '{filename}': {e}")
                return []

    def answer_user_question(self, context_for_llm, matched_node):
        prompt = f"""
You are a legal assistant AI trained to interpret and extract structured legal information from the Bharatiya Nyaya Sanhita (BNS) based on a legal knowledge graph.

Given the retrieved legal graph context and the user's target offense or concept, return the relevant:
- **Chapter Number and Name**
- **Section Number and Title**
- **Punishment Clause(s)**
- **Any Directly Related Offenses**

Ensure the answer is structured, cited exactly as in BNS, and only use information from the retrieved context.

---

### Example 1:

**Matched Node:** Theft

**Retrieved Context:**
Chapter XVII – Of Offenses Against Property
Section 303 – Theft
Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.

**Answer:**
- Chapter: Chapter XVII – Of Offenses Against Property
- Section: Section 303 – Theft
- Punishment: Imprisonment up to 3 years, or fine, or both
- Related Offenses: Attempt to commit theft, aggravated theft under Section 304

---

### Example 2:

**Matched Node:** Dacoity

**Retrieved Context:**
Chapter XVII – Of Offenses Against Property
Section 310 – Dacoity
When five or more persons conjointly commit or attempt to commit a robbery, it is called "dacoity". Punishment is imprisonment for life, or rigorous imprisonment for not less than 10 years.

**Answer:**
- Chapter: Chapter XVII – Of Offenses Against Property
- Section: Section 310 – Dacoity
- Punishment: Life imprisonment or rigorous imprisonment of not less than 10 years
- Related Offenses: Robbery (Section 309), preparation to commit dacoity (Section 311)

---

### User Query:

**Matched Node:** {matched_node}

**Retrieved Context:**
{context_for_llm}

---

### Answer:
"""

        response = self.llm.invoke(prompt)
        return response.content
