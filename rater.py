# rater.py - Multi-AI Rating Engine
# Rates AI output using Gemini and OpenAI APIs

import os
import json
import re

# Load API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

RATING_PROMPT_TEMPLATE = """You are a ruthless AI Quality Control Judge. Analyze the [User Query] and the [Base Model Output]. Your primary goal is FACTUAL ACCURACY and ERROR SEVERITY.

Rules:
1. FACT CHECK FIRST:
   - Identify factual errors, hallucinations, incorrect names, locations, or misleading claims.
   - Distinguish between:
     a) Critical errors (wrong facts, hallucinations) → heavily penalize (accuracy < 4)
     b) Minor issues (typos, phrasing) → small penalty
   - If the USER query itself contains errors, explicitly point them out separately.
2. GROUNDING: Evaluate whether the answer is verifiable and logically consistent.
3. CLARITY: Evaluate if the response is logically structured and easy to understand.
4. COMPLETENESS: Check if key aspects of the question are fully addressed.
5. SMART IMPROVEMENTS: Suggest only high-impact fixes - factual corrections, logic restructuring, missing key insights. NO generic suggestions.

[User Query]: {prompt}
[Base Model Output]: {output}

Respond ONLY in this exact JSON, nothing else:
{{"score": <avg of 3>, "accuracy": <0-10>, "clarity": <0-10>, "completeness": <0-10>, "fatal_flaw": "<main error or null>", "error_source": "<user|model|both>", "improvements": ["<fix1>", "<fix2>", "<fix3>"]}}"""

def get_mock_rating(base_score: float = 7.5) -> dict:
    """Return mock rating when API keys are missing"""
    return {
        "score": base_score,
        "accuracy": base_score,
        "clarity": base_score - 0.3,
        "completeness": base_score + 0.2,
        "improvements": [
            "Add more specific examples to illustrate key points",
            "Include edge cases and potential limitations",
            "Structure the response with clear headings"
        ]
    }

def parse_rating_response(text: str) -> dict:
    """
    Extract JSON from model response text
    Handles cases where model adds extra text around JSON
    """
    try:
        # Try direct JSON parse first
        return json.loads(text)
    except:
        # Try to find JSON block in text (greedy search to capture full JSON)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    # Return mock if parsing fails
    return get_mock_rating()

def rate_with_gemini(original_prompt: str, output: str) -> dict:
    """
    Rate AI response using Google Gemini Flash API
    Falls back to mock rating if API key missing or call fails
    """
    if not GEMINI_API_KEY:
        print("[RATER] Gemini API key missing, using mock rating")
        return get_mock_rating(7.8)
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        rating_prompt = RATING_PROMPT_TEMPLATE.format(
            prompt=original_prompt, 
            output=output
        )
        
        response = model.generate_content(rating_prompt)
        print("[RATER] Gemini rating received")
        return parse_rating_response(response.text)
        
    except Exception as e:
        print(f"[RATER] Gemini error: {str(e)}, using mock")
        return get_mock_rating(7.8)

def rate_with_openai(original_prompt: str, output: str) -> dict:
    """
    Rate AI response using OpenAI GPT-4o Mini API
    Falls back to mock rating if API key missing or call fails
    """
    if not OPENAI_API_KEY:
        print("[RATER] OpenAI API key missing, using mock rating")
        return get_mock_rating(7.3)
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        rating_prompt = RATING_PROMPT_TEMPLATE.format(
            prompt=original_prompt,
            output=output
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": rating_prompt}],
            temperature=0.3
        )
        
        print("[RATER] OpenAI rating received")
        return parse_rating_response(response.choices[0].message.content)
        
    except Exception as e:
        print(f"[RATER] OpenAI error: {str(e)}, using mock")
        return get_mock_rating(7.3)

def aggregate_ratings(gemini_rating: dict, openai_rating: dict) -> dict:
    """
    Combine Gemini and OpenAI ratings into final score
    Averages all metrics and merges improvement suggestions
    """
    final_score = round((gemini_rating["score"] + openai_rating["score"]) / 2, 1)
    
    # Combine unique improvements from both raters
    all_improvements = []
    seen = set()
    for tip in gemini_rating.get("improvements", []) + openai_rating.get("improvements", []):
        if tip not in seen:
            all_improvements.append(tip)
            seen.add(tip)
    
    return {
        "final_score": final_score,
        "top_improvements": all_improvements[:3]  # Top 3 only
    }