import time
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from router import classify_prompt
from ollama_client import generate, get_available_models
from rater import rate_with_gemini, rate_with_openai, aggregate_ratings
from pipeline import run_pipeline

# MODE: local_only = no API rating (default), hybrid = enable Gemini/OpenAI rating
MODE = "local_only"

app = FastAPI(title="Dolphax API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Cost config
COST_PER_1K = {
    "gpt-4o": 0.005,
    "gemini-1.5-pro": 0.0035,
    "claude-3-opus": 0.015,
    "local_ollama": 0.0
}

session_stats = {
    "total_tokens": 0,
    "total_saved_usd": 0.0,
    "queries_today": 0
}

# =========================
# Models
# =========================

class QueryRequest(BaseModel):
    prompt: str
    user_selected_category: str = "auto"  # auto, code, math, general, creative

class RegenerateRequest(BaseModel):
    prompt: str
    improvements: list[str]
    original_output: str

class PipelineRequest(BaseModel):
    prompt: str

# =========================
# Helpers
# =========================

def count_tokens(text: str) -> int:
    """
    Estimate token count (4 chars ≈ 1 token, common approximation).
    Used for cost calculation and stats tracking.
    """
    return max(1, len(text) // 4)

def check_intent_override(prompt: str, user_category: str) -> tuple[str, bool]:
    """
    Check if user's selected category matches prompt intent.
    Returns (final_category, was_overridden)
    """
    if user_category.lower() == "auto":
        # Use auto classification
        return (None, False)
    
    prompt_lower = prompt.lower()
    
    # Intent keywords for each category
    code_keywords = ["code", "build", "api", "function", "script", "write", "implement", "debug", "python", "javascript", "html", "css"]
    math_keywords = ["solve", "integral", "derivative", "equation", "calculate", "math", "algebra", "formula", "proof"]
    writing_keywords = ["write", "essay", "article", "story", "blog", "summarize", "explain", "describe"]
    
    detected_code = any(kw in prompt_lower for kw in code_keywords)
    detected_math = any(kw in prompt_lower for kw in math_keywords)
    detected_writing = any(kw in prompt_lower for kw in writing_keywords)
    
    # Check for mismatch
    if user_category.lower() == "code" and detected_code:
        return (user_category, False)
    elif user_category.lower() == "math" and detected_math:
        return (user_category, False)
    elif user_category.lower() == "writing" and detected_writing:
        return (user_category, False)
    elif user_category.lower() == "general" and not (detected_code or detected_math or detected_writing):
        return (user_category, False)
    elif user_category.lower() == "creative":
        return (user_category, False)  # Creative is always valid
    
    # Mismatch detected - override
    if detected_code:
        return ("code", True)
    elif detected_math:
        return ("math", True)
    else:
        return ("general", True)

# Category to model mapping (used in both /query and /regenerate)
CATEGORY_MODEL_MAP = {
    "code": "codellama:latest",
    "math": "deepseek-r1:latest",
    "writing": "qwen3:latest",
    "creative": "qwen3:latest",
    "general": "qwen3:latest",
}

def calculate_real_savings(prompt: str, output: str) -> dict:
    prompt_tokens = count_tokens(prompt)
    output_tokens = count_tokens(output)
    total_tokens = prompt_tokens + output_tokens

    gpt4o_cost = (total_tokens / 1000) * COST_PER_1K["gpt-4o"]
    saved_usd = round(gpt4o_cost, 6)

    session_stats["total_tokens"] += total_tokens
    session_stats["total_saved_usd"] += saved_usd
    session_stats["queries_today"] += 1

    return {
        "total_tokens": total_tokens,
        "saved_usd": saved_usd,
        "session_total_tokens": session_stats["total_tokens"],
        "session_saved_usd": round(session_stats["total_saved_usd"], 6),
        "queries_today": session_stats["queries_today"]
    }

def optimize_prompt(raw_prompt: str, category: str) -> str:
    """
    Convert raw user prompt into category-specific optimized prompt.
    Routes to three modes: CODE, GENERAL (for real-world topics), or CREATIVE (for ideas).
    """
    prompt_lower = raw_prompt.lower()
    
    if category == "code":
        return f"""You are an expert programmer.

Task: {raw_prompt}

Rules:
- Return ONLY code (no explanations unless asked)
- Use clear variable names and proper formatting
- Include helpful comments for complex logic
- If HTML/CSS/JS requested: return ONLY that language
- If Python/JavaScript/other: return production-ready code

Optimize for clarity and correctness."""
    
    elif category == "math":
        return f"""You are a math expert.

Task: {raw_prompt}

Rules:
- Show step-by-step working for clarity
- Label each step clearly
- Give the final answer at the end
- Use proper mathematical notation
- If a formula applies, show it

Be precise and thorough."""
    
    elif category == "writing":
        return f"""You are a clear communicator.

Task: {raw_prompt}

Rules:
- Structure your response clearly (use sections if helpful)
- Be concise but thorough
- Use examples where appropriate
- Adapt your tone to the topic

Write clearly and helpfully."""
    
    else:  # general
        # Detect CREATIVE MODE: ideas, innovation, startups, "crazy" concepts
        creative_keywords = [
            "idea", "startup idea", "ai idea", "creative", "innovation", "invent",
            "brainstorm", "concept", "feature", "product idea", "business idea",
            "app idea", "what if", "imagine", "create", "build", "generate idea",
            "bold idea", "new idea", "unique", "novel", "disruptive", "crazy idea",
            "wild idea", "future", "could we", "would you"
        ]
        is_creative_mode = any(kw in prompt_lower for kw in creative_keywords)
        
        if is_creative_mode:
            return f"""You are a bold, visionary product strategist and innovator.

Task: {raw_prompt}

GENERATE ORIGINAL IDEAS. Always provide at least one real, specific idea.

Structure each idea:
1. **Idea Name** (catchy, memorable)
2. **What It Does** (1–2 sentences, crystal clear value)
3. **Why It Matters** (the real insight or problem solved)
4. **Key Feature(s)** (one bold differentiator that makes it work)

CRITICAL RULES:
- Ideas must feel like REAL PRODUCTS, not generic advice
- Avoid filler: NO "you can try", NO "for example", NO vague suggestions
- Be specific, confident, and slightly futuristic
- Non-obvious thinking—no textbook concepts
- Keep concise but impactful (no rambling)
- Every detail should matter

Generate bold, actionable, memorable ideas."""
        
        # Detect GENERAL MODE: companies, universities, brands, organizations, real-world topics
        general_keywords = [
            "company", "brand", "organization", "startup", "product", "corporation",
            "service", "platform", "tool", "software", "university", "college",
            "institute", "school", "business", "firm", "enterprise", "team",
            "what is", "who is", "tell me about", "explain", "describe"
        ]
        is_general_mode = any(kw in prompt_lower for kw in general_keywords)
        
        if is_general_mode:
            return f"""You are a confident, knowledgeable knowledge assistant.

Task: {raw_prompt}

GENERAL MODE: Real-world topics, companies, organizations, universities, brands.

Assumptions & Rules:
- ASSUME this entity likely exists unless clearly fictional
- Provide a HELPFUL EXPLANATION instead of refusing or saying "not recognized"
- Be CONFIDENT even if uncertain—base answers on available knowledge

Response Structure:
1. **Summary** (1–2 sentences: what it is/does)
2. **Key Points** (2–3 bullets: what it does, why it's known, key characteristics)
3. **Note** (if uncertain, add: "based on available knowledge, ..." but still answer)

Critical Rules:
- NEVER refuse without trying
- NEVER say "I don't have information" and stop
- Be clear, practical, and intentional
- Make the answer useful and confident

Always provide a helpful answer."""
        
        # DEFAULT: General helpful response
        else:
            return f"""You are a helpful, clear assistant.

Task: {raw_prompt}

Rules:
- Answer directly and clearly
- Provide context where helpful
- Be accurate, thoughtful, and useful
- Structure your response for readability

Help the user effectively."""

def is_output_weak(output: str, category: str) -> bool:
    """
    Lightweight quality check without calling Ollama.
    Detects obviously weak or incomplete outputs.
    """
    if not output or output.startswith("Error"):
        return True
    
    # Very short output (< 40 chars, not counting whitespace)
    if len(output.strip()) < 40:
        return True
    
    # Says "I don't know" without trying to help
    lower = output.lower()
    if "i don't know" in lower and len(output.strip()) < 100:
        return True
    
    # Incomplete response (ends with ellipsis or "to be continued")
    if output.rstrip().endswith("...") or "to be continued" in lower:
        return True
    
    # For code, check if it's obviously incomplete
    if category == "code":
        if output.count("{") != output.count("}"):
            return True
        if output.count("[") != output.count("]"):
            return True
    
    return False

def detect_refusal_or_vague_answer(output: str) -> bool:
    """
    Detect if model refused to answer or gave a vague non-answer.
    Looks for phrases indicating the model doesn't know or won't help.
    """
    lower = output.lower()
    
    # Refusal and uncertainty phrases
    refusal_phrases = [
        "not recognized",
        "no information",
        "i don't have information",
        "unknown to me",
        "cannot find",
        "no record",
        "not aware",
        "cannot confirm",
        "don't have access",
        "not in my knowledge",
        "unable to find",
        "cannot provide",
        "i'm not familiar",
        "not documented"
    ]
    
    # Check for refusal phrases
    if any(phrase in lower for phrase in refusal_phrases):
        return True
    
    # Check if output is suspiciously vague (too short even for general queries)
    if len(output.strip()) < 60 and any(phrase in lower for phrase in ["i'm", "it is", "they are"]):
        return True
    
    return False

def create_fallback_entity_prompt(raw_prompt: str) -> str:
    """
    Create a stronger prompt for real-world entities that assumes they exist.
    Used when model refuses or gives vague answer.
    """
    return f"""You are a confident knowledge assistant.

Task: {raw_prompt}

IMPORTANT: This is a real-world entity that exists. Provide the best possible explanation based on available knowledge.

Structure:
1. What is it? (1-2 sentences)
2. What does it do? (2-3 bullet points)
3. Why is it known? (1 sentence, optional)

Be confident and helpful. Do not refuse to answer. Provide your best knowledge-based response."""

def improve_weak_output(optimized_prompt: str, previous_output: str, category: str) -> str:
    """
    Create an improved prompt to regenerate weak output.
    Asks for more detail and completeness.
    """
    if category == "code":
        return f"""{optimized_prompt}

The previous attempt was incomplete. Please provide:
- Complete, runnable code
- All necessary imports and dependencies
- Proper structure and formatting
- Add comments explaining key sections

Generate a thorough, complete solution."""
    
    else:
        return f"""{optimized_prompt}

The previous attempt was too brief. Please:
- Provide more detail and explanation
- Include examples or specific information
- Make the answer more complete and helpful
- Address all aspects of the question

Generate a thorough, detailed response."""

def build_enhanced_prompt(user_prompt: str) -> str:
    return f"""
You are an expert programmer.

Rules:
- If user asks for HTML → return ONLY HTML code
- CSS → return ONLY CSS
- JavaScript → return ONLY JS
- Python → return ONLY Python
- No explanations unless asked

User request:
{user_prompt}
"""

def compare_outputs(original: str, improved: str) -> dict:
    orig_tokens = count_tokens(original)
    impr_tokens = count_tokens(improved)

    length_ratio = round(impr_tokens / max(1, orig_tokens), 2)

    return {
        "original_tokens": orig_tokens,
        "improved_tokens": impr_tokens,
        "length_ratio": length_ratio,
        "verdict": "Better" if length_ratio > 1 else "Similar"
    }

# =========================
# Routes
# =========================

@app.post("/query")
async def handle_query(request: QueryRequest):
    start_time = time.time()

    # Step 0: Check for user category override
    user_selected = request.user_selected_category.lower() if request.user_selected_category else "auto"
    
    if user_selected == "auto":
        # Use automatic classification
        routing = classify_prompt(request.prompt)
        final_category = routing["category"]
        category_adjusted = False
    else:
        # Check if user selection matches prompt intent
        override_category, was_overridden = check_intent_override(request.prompt, user_selected)
        routing = classify_prompt(request.prompt)
        
        if override_category is not None:
            # Mismatch detected or override returned a category
            final_category = override_category
            category_adjusted = was_overridden
        else:
            # User selection is valid - use it
            final_category = user_selected
            category_adjusted = False
    
    print(f"[DOLPHAX] User selected: {user_selected}, Final category: {final_category}")
    print(f"[DOLPHAX] Classification: {routing['category']}")
    
    # Get model from CATEGORY_MODEL_MAP
    model_to_use = CATEGORY_MODEL_MAP.get(final_category, "qwen3:latest")
    confidence_score = routing["confidence_score"]

    # Step 2: Optimize the prompt using final_category
    optimized_prompt = optimize_prompt(request.prompt, final_category)
    print(f"[DOLPHAX] Prompt optimized for category: {final_category}")

    # Step 3: Generate answer
    output = generate(model_to_use, optimized_prompt)
    elapsed = round(time.time() - start_time, 2)
    print(f"[DOLPHAX] Output received from {model_to_use} in {elapsed}s")
    
    # Handle Ollama errors
    if isinstance(output, str) and output.startswith("Error"):
        elapsed_time = round(time.time() - start_time, 2)
        return {
            "error": output,
            "raw_prompt": request.prompt,
            "optimized_prompt": optimized_prompt,
            "user_selected_category": user_selected,
            "final_category_used": final_category,
            "category_adjusted": category_adjusted,
            "model_used": model_to_use,
            "confidence": confidence_score,
            "processing_time": elapsed_time
        }

    # Step 4: Quality check - detect incomplete outputs
    improved_once = False
    fallback_once = False
    quality_note = "Output looks good"
    
    if MODE == "local_only":
        # Quality check 1: Weak outputs (incomplete code, too short, etc)
        if is_output_weak(output, final_category):
            print(f"[DOLPHAX] Output quality weak, improving once...")
            improved_prompt = improve_weak_output(optimized_prompt, output, final_category)
            output = generate(model_to_use, improved_prompt)
            improved_once = True
            quality_note = "Output was incomplete, improved and regenerated"
        
        # Quality check 2: Refusal/vague answers (especially for real-world entities)
        if detect_refusal_or_vague_answer(output):
            print(f"[DOLPHAX] Output is refusal or vague, using fallback prompt...")
            fallback_prompt = create_fallback_entity_prompt(request.prompt)
            output = generate(model_to_use, fallback_prompt)
            fallback_once = True
            quality_note = "Output was vague/refused, regenerated with confident prompt"
    
    # Step 5: Calculate savings
    savings = calculate_real_savings(request.prompt, output)
    elapsed_time = round(time.time() - start_time, 2)
    print(f"[DOLPHAX] Request complete in {elapsed_time}s, tokens saved: ${savings['saved_usd']}")

    # Step 6: Build response
    adjustment_note = ""
    if category_adjusted:
        adjustment_note = f"Adjusted to {final_category} based on prompt intent"
    
    response = {
        "raw_prompt": request.prompt,
        "optimized_prompt": optimized_prompt,
        "user_selected_category": user_selected,
        "final_category_used": final_category,
        "category_adjusted": category_adjusted,
        "adjustment_note": adjustment_note,
        "model_used": model_to_use,
        "confidence": confidence_score,
        "output": output,
        "quality_check": quality_note,
        "improved_once": improved_once or fallback_once,
        "processing_time": elapsed_time,
        "token_stats": savings,
    }

    # Add ratings if in hybrid mode
    if MODE == "hybrid":
        print(f"[DOLPHAX] Hybrid mode: requesting API ratings...")
        gemini_rating = rate_with_gemini(request.prompt, output)
        openai_rating = rate_with_openai(request.prompt, output)
        aggregated = aggregate_ratings(gemini_rating, openai_rating)
        response["final_score"] = aggregated["final_score"]
        response["top_improvements"] = aggregated["top_improvements"]

    return response

@app.post("/regenerate")
async def regenerate(request: RegenerateRequest):
    start_time = time.time()

    improvements_text = "\n".join(f"- {imp}" for imp in request.improvements)

    # Classify the prompt to get category
    routing = classify_prompt(request.prompt)
    final_category = routing["category"]
    
    # Optimize prompt using the correct category
    optimized_prompt = optimize_prompt(request.prompt, final_category)
    
    # Build regeneration prompt on top of the optimized prompt
    enhanced_prompt = f"""{optimized_prompt}

Please improve the response using these points:
{improvements_text}

Provide a thorough, improved response."""

    # Get model from CATEGORY_MODEL_MAP
    model_to_use = CATEGORY_MODEL_MAP.get(final_category, "qwen3:latest")

    new_output = generate(model_to_use, enhanced_prompt)
    
    # Check if Ollama returned an error string
    if isinstance(new_output, str) and new_output.startswith("Error"):
        elapsed_time = round(time.time() - start_time, 2)
        print(f"[DOLPHAX] Output received in {elapsed_time}s")
        return {
            "error": new_output,
            "processing_time": elapsed_time
        }

    # Skip rating based on MODE
    if MODE == "local_only":
        elapsed_time = round(time.time() - start_time, 2)
        print(f"[DOLPHAX] Output received in {elapsed_time}s")
        print(f"[DOLPHAX] Rating skipped - local_only mode")
        
        savings = calculate_real_savings(request.prompt, new_output)
        comparison = compare_outputs(request.original_output, new_output)
        
        return {
            "output": new_output,
            "processing_time": elapsed_time,
            "token_stats": savings,
            "comparison": comparison,
            "note": "Rating skipped - running in local_only mode"
        }
    
    # Hybrid mode: enable API ratings
    gemini_rating = rate_with_gemini(request.prompt, new_output)
    openai_rating = rate_with_openai(request.prompt, new_output)
    aggregated = aggregate_ratings(gemini_rating, openai_rating)

    savings = calculate_real_savings(request.prompt, new_output)

    comparison = compare_outputs(request.original_output, new_output)
    
    elapsed_time = round(time.time() - start_time, 2)
    print(f"[DOLPHAX] Output received in {elapsed_time}s")

    return {
        "output": new_output,
        "final_score": aggregated["final_score"],
        "top_improvements": aggregated["top_improvements"],
        "processing_time": elapsed_time,
        "token_stats": savings,
        "comparison": comparison
    }

@app.post("/api/pipeline")
async def pipeline_endpoint(request: PipelineRequest):
    """
    NEW: 6-Stage Dolphax AI Pipeline
    
    Stages:
    1. PROMPT ENHANCER - rewrites input to be precise (<150 words)
    2. TASK CLASSIFIER - detects task type, complexity, expected tokens
    3. MODEL SELECTOR - picks ONE best model + different verifier
    4. EXECUTOR - runs task with selected model
    5. OUTPUT VERIFIER - different model scores output (0-100)
    6. IMPROVER - fixes flagged issues if score < 90
    
    Returns: Complete pipeline execution trace with all stages and final output
    """
    start_time = time.time()
    
    print(f"[PIPELINE] Starting 6-stage pipeline for prompt: {request.prompt[:50]}...")
    
    try:
        # Run the complete 6-stage pipeline
        pipeline_result = run_pipeline(request.prompt)
        
        elapsed_time = round(time.time() - start_time, 2)
        
        # Calculate token savings
        if pipeline_result["status"] == "completed":
            final_output = pipeline_result["final_output"]
            savings = calculate_real_savings(request.prompt, final_output)
            pipeline_result["token_stats"] = savings
        
        pipeline_result["total_processing_time"] = elapsed_time
        
        print(f"[PIPELINE] Pipeline complete in {elapsed_time}s")
        print(f"[PIPELINE] Final status: {pipeline_result['verification_status']}")
        
        return pipeline_result
        
    except Exception as e:
        import traceback
        elapsed_time = round(time.time() - start_time, 2)
        
        print(f"[PIPELINE] ERROR: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time": elapsed_time
        }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open("../frontend/dolphax.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h2>dolphax.html not found next to main.py</h2>", status_code=404)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "models": get_available_models(),
        "session_stats": session_stats
    }