# router.py - Smart Prompt Classifier with Creative Detection

def classify_prompt(prompt: str) -> dict:
    prompt_lower = prompt.lower()

    # Code keywords - programming specific (refined to avoid false positives)
    code_keywords = [
        "code", "function", "bug", "error", "program", "script", "debug",
        "python", "javascript", "html", "css", "react", "frontend",
        "array", "string", "list", "loop",
        "class", "object", "return", "variable", "algorithm", "leetcode",
        "indices", "index", "implement", "solution", "solve this code",
        "nums", "pointer", "stack", "queue", "tree", "graph",
        "complexity", "runtime", "output", "input array", "integers array",
        "given an array", "given a string", "hashmap", "dictionary",
        "import", "def ", "console.log", "compile", "syntax error"
    ]

    # Math keywords - pure math only
    math_keywords = [
        "calculate", "equation", "formula", "integral", "derivative",
        "trigonometry", "algebra", "geometry", "matrix", "determinant",
        "differentiate", "probability", "statistics", "calculus",
        "arithmetic", "percentage", "what is 2+2", "solve for x"
    ]

    # Writing keywords
    writing_keywords = [
        "write", "essay", "story", "poem", "article", "summarize",
        "explain", "describe", "blog", "email", "letter"
    ]

    # Creative keywords - innovation, brainstorm, ideas
    creative_keywords = [
        "brainstorm", "idea", "startup idea", "ai idea", "creative", "innovation",
        "invent", "innovate", "concept", "feature", "product idea", "business idea",
        "app idea", "what if", "imagine", "generate idea", "bold idea", "new idea",
        "unique", "novel", "disruptive", "crazy idea", "wild idea", "future"
    ]

    # Score calculation
    code_score = sum(1 for word in code_keywords if word in prompt_lower)
    math_score = sum(1 for word in math_keywords if word in prompt_lower)
    writing_score = sum(1 for word in writing_keywords if word in prompt_lower)
    creative_score = sum(1 for word in creative_keywords if word in prompt_lower)

    # CODE gets priority over math if both match
    # because coding problems often mention numbers/integers
    if code_score > 0 and code_score >= math_score and code_score >= creative_score:
        category = "code"
        model = "codellama:latest"
        confidence = min(0.95, 0.6 + (code_score * 0.1))

    elif math_score > code_score and math_score > creative_score:
        category = "math"
        model = "deepseek-r1:latest"
        confidence = min(0.95, 0.6 + (math_score * 0.1))

    elif creative_score > 0 and creative_score > code_score and creative_score > math_score:
        category = "creative"
        model = "qwen3:latest"
        confidence = min(0.95, 0.6 + (creative_score * 0.1))

    elif writing_score > 0:
        category = "writing"
        model = "qwen3:latest"
        confidence = min(0.95, 0.6 + (writing_score * 0.1))

    else:
        category = "general"
        model = "qwen3:latest"
        confidence = 0.5

    print(f"[DOLPHAX] Routing to: {model}")
    print(f"[ROUTER] Scores — Code: {code_score}, Math: {math_score}, Writing: {writing_score}, Creative: {creative_score}")

    return {
        "category": category,
        "model": model,
        "confidence_score": confidence
    }