# router.py - Smart Prompt Classifier with Model Selector

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


# ============================================================================
# STAGE 3: MODEL SELECTOR - Enhanced for Pipeline
# ============================================================================
# This function is used by the pipeline orchestrator for intelligent model selection
# Priority: Accuracy → Token Efficiency → Speed (when accuracy_critical=False)

def select_best_model(task_type: str, complexity: str, accuracy_critical: bool) -> dict:
    """
    STAGE 3 - MODEL SELECTOR: Select ONE best model + different verifier model
    
    Model capabilities:
    - codellama:latest: Coding (accuracy: 9/10, speed: 7/10, efficiency: 8/10)
    - deepseek-r1:latest: Math/Reasoning (accuracy: 9/10, speed: 5/10, efficiency: 7/10)
    - qwen3:latest: General/Creative/Writing (accuracy: 8/10, speed: 8/10, efficiency: 8/10)
    
    Selection logic:
    - If accuracy_critical: Pick highest accuracy model for task type
    - Otherwise: Balance efficiency and speed
    - Verifier model is ALWAYS different from selected model
    """
    
    # Model profiles
    models_db = {
        "codellama:latest": {
            "best_for": ["coding"],
            "accuracy": 9,
            "speed": 7,
            "efficiency": 8,
            "description": "Code generation and programming tasks"
        },
        "deepseek-r1:latest": {
            "best_for": ["math", "reasoning", "summarization", "research"],
            "accuracy": 9,
            "speed": 5,
            "efficiency": 7,
            "description": "Deep reasoning and mathematical tasks"
        },
        "qwen3:latest": {
            "best_for": ["general", "creative", "writing", "classification"],
            "accuracy": 8,
            "speed": 8,
            "efficiency": 8,
            "description": "General tasks, creative writing, and classification"
        }
    }
    
    # Decision tree for model selection
    if accuracy_critical:
        # ACCURACY PRIORITY: Choose most accurate model for this task
        if task_type == "coding":
            selected_model = "codellama:latest"
            # Verifier must be different: pick deepseek or qwen
            verifier_model = "deepseek-r1:latest" if complexity == "complex" else "qwen3:latest"
        
        elif task_type in ["math", "reasoning"]:
            selected_model = "deepseek-r1:latest"
            # Verifier must be different: pick codellama or qwen
            verifier_model = "codellama:latest" if "code" in task_type else "qwen3:latest"
        
        else:  # general, creative, writing, classification, research, summarization
            selected_model = "qwen3:latest"
            # Verifier must be different: pick deepseek for complex, codellama otherwise
            verifier_model = "deepseek-r1:latest" if complexity == "complex" else "codellama:latest"
    
    else:
        # EFFICIENCY + SPEED PRIORITY: Balance token efficiency and response speed
        if task_type == "coding":
            selected_model = "codellama:latest"
            verifier_model = "qwen3:latest"
        
        else:  # For other tasks, qwen3 is faster and efficient
            selected_model = "qwen3:latest"
            # Verifier: alternate between deepseek and codellama
            verifier_model = "deepseek-r1:latest" if complexity == "complex" else "codellama:latest"
    
    reasoning = (
        f"Selected '{selected_model}' for {task_type} ({complexity} complexity, "
        f"accuracy_critical={accuracy_critical}). Using '{verifier_model}' for verification."
    )
    
    print(f"[ROUTER] MODEL SELECTOR: {reasoning}")
    
    return {
        "selected_model": selected_model,
        "verifier_model": verifier_model,
        "reasoning": reasoning,
        "selected_model_info": models_db[selected_model],
        "verifier_model_info": models_db[verifier_model]
    }