import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
#genai.configure(api_key=os.getenv(GEMINI_API_KEY))
genai.configure(api_key="AIzaSyDEQokgWPkgqA5WKNXZPMYxhsVaDC-NTkk")
model = genai.GenerativeModel("gemini-1.5-flash")

def summarize_report(chunks):
    full_context = "\n".join(chunks)
    prompt = f"""
A patient has uploaded the following medical report:

---
{full_context}
---

Please write a **friendly and easy-to-understand summary** of this medical report for a person with no medical background.

Include:
1. Which results are normal.
2. Which are abnormal and what they might indicate.
3. Possible causes or implications.
4. A polite suggestion to consult a doctor.
"""

    response = model.generate_content(prompt)
    return response.text
