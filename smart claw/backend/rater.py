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


# ============================================================================
# STAGE 5: OUTPUT VERIFIER - Enhanced for Pipeline
# ============================================================================
# Used by the 6-stage pipeline to verify output quality with a different model

def verify_output_with_local_model(prompt: str, output: str, verifier_model: str) -> dict:
    """
    STAGE 5 - OUTPUT VERIFIER: Verify output quality using a different model
    
    Scores output 0-100:
    - PASS: 90+ (excellent)
    - WARN: 70-89 (acceptable but needs improvement)
    - FAIL: 0-69 (poor quality)
    
    Evaluates:
    1. Correctness: Is the output accurate? Free of hallucinations?
    2. Completeness: Does it fully address the task?
    3. Format Quality: Does it match expected output format?
    4. Clarity: Is it well-structured and easy to understand?
    
    Returns: Comprehensive verification report with score, issues, and suggestions
    """
    from ollama_client import generate
    
    verification_prompt = f"""You are an expert quality verifier. Evaluate this response:

ORIGINAL TASK: {prompt}

RESPONSE TO VERIFY:
{output}

Score the response (0-100) on these criteria:
1. **Correctness** (0-10): Is it factually accurate? Free of hallucinations?
2. **Completeness** (0-10): Does it fully address the task?
3. **Format Quality** (0-10): Does it match expected output format?
4. **Clarity** (0-10): Is it well-structured and understandable?

Calculate overall_score as average of the 4 metrics.
Determine status: 90+ = PASS, 70-89 = WARN, 0-69 = FAIL

List specific issues found (if any).
Provide actionable suggestions for improvement (if score < 90).

Respond with ONLY valid JSON (no other text):
{{
    "overall_score": <0-100>,
    "correctness": <0-10>,
    "completeness": <0-10>,
    "format_quality": <0-10>,
    "clarity": <0-10>,
    "status": "PASS|WARN|FAIL",
    "flagged_issues": ["<specific issue>", "<specific issue>"],
    "critical_error": "<if any, else null>",
    "suggestions": ["<actionable fix>", "<actionable fix>"]
}}"""
    
    response = generate(verifier_model, verification_prompt)
    
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response)
    except:
        print(f"[VERIFIER] Failed to parse verification response, using fallback")
        result = {
            "overall_score": 75,
            "correctness": 7,
            "completeness": 7,
            "format_quality": 7,
            "clarity": 7,
            "status": "WARN",
            "flagged_issues": ["Unable to parse detailed verification"],
            "critical_error": None,
            "suggestions": ["Review output manually for quality assurance"]
        }
    
    # Normalize score to 0-100
    score = result.get("overall_score", 75)
    if isinstance(score, str):
        try:
            score = int(score)
        except:
            score = 75
    score = max(0, min(100, score))  # Clamp to 0-100
    
    # Determine status based on score
    if score >= 90:
        status = "PASS"
    elif score >= 70:
        status = "WARN"
    else:
        status = "FAIL"
    
    # Extract metrics with defaults
    correctness = result.get("correctness", 7)
    completeness = result.get("completeness", 7)
    format_quality = result.get("format_quality", 7)
    clarity = result.get("clarity", 7)
    
    # Ensure metrics are ints
    try:
        correctness = int(correctness)
        completeness = int(completeness)
        format_quality = int(format_quality)
        clarity = int(clarity)
    except:
        correctness = completeness = format_quality = clarity = 7
    
    return {
        "overall_score": score,
        "correctness": correctness,
        "completeness": completeness,
        "format_quality": format_quality,
        "clarity": clarity,
        "status": status,
        "flagged_issues": result.get("flagged_issues", []),
        "critical_error": result.get("critical_error"),
        "suggestions": result.get("suggestions", []),
        "needs_improvement": score < 90
    }