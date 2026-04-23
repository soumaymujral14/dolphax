# pipeline.py - Dolphax 6-Stage AI Pipeline Orchestrator
# Stages: Prompt Enhancer → Task Classifier → Model Selector → Executor → Output Verifier → Improver

import json
import re
from typing import Dict, List, Tuple

try:
    from .ollama_client import generate, count_tokens_local
    from .router import classify_prompt, select_best_model
    from .rater import verify_output_with_local_model
except ImportError:
    from ollama_client import generate, count_tokens_local
    from router import classify_prompt, select_best_model
    from rater import verify_output_with_local_model


# ============================================================================
# STAGE 1: PROMPT ENHANCER - Rewrites raw input to be precise & token-efficient
# ============================================================================

def stage1_enhance_prompt(raw_prompt: str) -> Dict:
    """
    Stage 1: PROMPT ENHANCER
    - Removes redundancy and filler
    - Ensures clarity and precision
    - Suggests expected output format
    - Token budget: <200 output
    """
    
    enhancer_prompt = f"""You are a prompt optimization expert. Rewrite this prompt to be:
1. Precise and clear (no filler words)
2. Under 150 words
3. Include expected output format if applicable

Original prompt: {raw_prompt}

Respond with ONLY valid JSON (no other text):
{{
    "enhanced_prompt": "<rewritten prompt under 150 words>",
    "output_format": "<describe expected output: code/text/list/json/etc>",
    "detected_context": "<key info inferred from prompt>"
}}"""
    
    response = generate("qwen3:latest", enhancer_prompt)
    
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response)
    except:
        # Fallback if parsing fails
        result = {
            "enhanced_prompt": raw_prompt[:150],
            "output_format": "text",
            "detected_context": "Unable to parse"
        }
    
    return {
        "stage": "enhancer",
        "status": "success",
        "raw_prompt": raw_prompt,
        "enhanced_prompt": result.get("enhanced_prompt", raw_prompt),
        "output_format": result.get("output_format", "text"),
        "detected_context": result.get("detected_context", ""),
        "tokens_used": count_tokens_local(response)
    }


# ============================================================================
# STAGE 2: TASK CLASSIFIER - Detects task type, complexity, expected tokens
# ============================================================================

def stage2_classify_task(enhanced_prompt: str, output_format: str) -> Dict:
    """
    Stage 2: TASK CLASSIFIER
    - Detects: task_type, complexity, expected_output_tokens
    - Task types: coding, math, reasoning, summarization, research, creative, classification
    - Complexity: simple (1-2 steps), medium (3-5 steps), complex (5+ steps)
    - Token budget: <100 output
    """
    
    classifier_prompt = f"""Analyze this task and classify it. Return ONLY valid JSON:

Task: {enhanced_prompt}
Expected output format: {output_format}

Return JSON:
{{
    "task_type": "coding|math|reasoning|summarization|research|creative|classification",
    "complexity": "simple|medium|complex",
    "description": "<brief explanation>",
    "expected_output_tokens": <number between 200-2000>,
    "reasoning_required": <true/false>,
    "accuracy_critical": <true/false>
}}"""
    
    response = generate("qwen3:latest", classifier_prompt)
    
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        result = json.loads(json_match.group() if json_match else response)
    except:
        result = {
            "task_type": "reasoning",
            "complexity": "medium",
            "expected_output_tokens": 500,
            "accuracy_critical": True
        }
    
    return {
        "stage": "classifier",
        "status": "success",
        "task_type": result.get("task_type", "reasoning"),
        "complexity": result.get("complexity", "medium"),
        "expected_output_tokens": result.get("expected_output_tokens", 500),
        "accuracy_critical": result.get("accuracy_critical", True),
        "description": result.get("description", ""),
        "tokens_used": count_tokens_local(response)
    }


# ============================================================================
# STAGE 3: MODEL SELECTOR - Picks ONE best model with separate verifier
# ============================================================================

def stage3_select_model(task_type: str, complexity: str, accuracy_critical: bool) -> Dict:
    """
    Stage 3: MODEL SELECTOR
    - Uses enhanced router.select_best_model function
    - Selects ONE model based on: accuracy > token_efficiency > speed
    - Always picks different model for verifier
    - Token budget: <100 output (no actual generation)
    """
    
    # Call the router's enhanced model selector
    result = select_best_model(task_type, complexity, accuracy_critical)
    
    return {
        "stage": "model_selector",
        "status": "success",
        "selected_model": result["selected_model"],
        "verifier_model": result["verifier_model"],
        "reasoning": result["reasoning"],
        "selected_model_info": result["selected_model_info"],
        "verifier_model_info": result["verifier_model_info"],
        "tokens_used": 0  # No actual generation, pure logic
    }


# ============================================================================
# STAGE 4: EXECUTOR - Runs task with selected model
# ============================================================================

def stage4_execute_task(selected_model: str, enhanced_prompt: str, expected_output_tokens: int) -> Dict:
    """
    Stage 4: EXECUTOR
    - Runs the task using ollama_client
    - Uses selected model
    - Token budget: <2000 output
    """
    
    # Add task execution context
    execution_prompt = f"""{enhanced_prompt}

Provide a clear, accurate response."""
    
    try:
        output = generate(selected_model, execution_prompt)
        tokens_used = count_tokens_local(output)
        
        return {
            "stage": "executor",
            "status": "success",
            "model_used": selected_model,
            "raw_output": output,
            "tokens_used": tokens_used,
            "within_budget": tokens_used <= 2000
        }
    except Exception as e:
        return {
            "stage": "executor",
            "status": "error",
            "model_used": selected_model,
            "raw_output": f"Error: {str(e)}",
            "tokens_used": 0,
            "within_budget": False
        }


# ============================================================================
# STAGE 5: OUTPUT VERIFIER - Verifies output with different model
# ============================================================================

def stage5_verify_output(enhanced_prompt: str, raw_output: str, verifier_model: str) -> Dict:
    """
    Stage 5: OUTPUT VERIFIER
    - Uses different model to cross-check output
    - Scores 0-100: PASS=90+, WARN=70-89, FAIL=0-69
    - Evaluates: correctness, completeness, hallucination, format
    - Token budget: <200 output
    """
    
    # Use the enhanced verifier from rater.py
    verification_result = verify_output_with_local_model(enhanced_prompt, raw_output, verifier_model)
    
    # Add stage metadata
    verification_result["stage"] = "verifier"
    verification_result["status"] = "success"
    verification_result["model_used"] = verifier_model
    verification_result["tokens_used"] = count_tokens_local(raw_output) // 2  # Rough estimate
    
    return verification_result


# ============================================================================
# STAGE 6: IMPROVER - Fixes flagged issues only (fires if score < 90)
# ============================================================================

def stage6_improve_output(raw_output: str, enhanced_prompt: str, 
                         flagged_issues: List[str], selected_model: str) -> Dict:
    """
    Stage 6: IMPROVER
    - Only fires if verifier score < 90
    - Fixes ONLY flagged issues
    - One pass only
    - Uses same model as executor
    """
    
    issues_text = "\n".join([f"- {issue}" for issue in flagged_issues])
    
    improver_prompt = f"""You are an expert improver. This response had issues:

ORIGINAL TASK: {enhanced_prompt}

CURRENT RESPONSE:
{raw_output}

ISSUES TO FIX:
{issues_text}

Fix ONLY these issues. Keep the rest of the response unchanged. Provide improved response:"""
    
    try:
        improved = generate(selected_model, improver_prompt)
        tokens_used = count_tokens_local(improved)
        
        return {
            "stage": "improver",
            "status": "success",
            "model_used": selected_model,
            "improved_output": improved,
            "tokens_used": tokens_used,
            "issues_fixed": len(flagged_issues),
            "within_budget": tokens_used <= 2000
        }
    except Exception as e:
        return {
            "stage": "improver",
            "status": "error",
            "model_used": selected_model,
            "improved_output": raw_output,
            "tokens_used": 0,
            "error": str(e)
        }


# ============================================================================
# ORCHESTRATOR - Runs all 6 stages in sequence
# ============================================================================

def run_pipeline(raw_prompt: str) -> Dict:
    """
    ORCHESTRATOR: Runs all 6 stages in sequence
    Returns complete pipeline execution trace
    """
    
    pipeline_trace = {
        "status": "running",
        "stages": {},
        "final_output": None,
        "verification_status": None,
        "total_tokens": 0
    }
    
    try:
        # Stage 1: Enhance prompt
        print("[PIPELINE] Stage 1: Prompt Enhancer")
        stage1 = stage1_enhance_prompt(raw_prompt)
        pipeline_trace["stages"]["enhancer"] = stage1
        enhanced_prompt = stage1["enhanced_prompt"]
        output_format = stage1["output_format"]
        pipeline_trace["total_tokens"] += stage1["tokens_used"]
        
        # Stage 2: Classify task
        print("[PIPELINE] Stage 2: Task Classifier")
        stage2 = stage2_classify_task(enhanced_prompt, output_format)
        pipeline_trace["stages"]["classifier"] = stage2
        task_type = stage2["task_type"]
        complexity = stage2["complexity"]
        accuracy_critical = stage2["accuracy_critical"]
        pipeline_trace["total_tokens"] += stage2["tokens_used"]
        
        # Stage 3: Select model
        print("[PIPELINE] Stage 3: Model Selector")
        stage3 = stage3_select_model(task_type, complexity, accuracy_critical)
        pipeline_trace["stages"]["model_selector"] = stage3
        selected_model = stage3["selected_model"]
        verifier_model = stage3["verifier_model"]
        
        # Stage 4: Execute
        print("[PIPELINE] Stage 4: Executor")
        stage4 = stage4_execute_task(selected_model, enhanced_prompt, stage2["expected_output_tokens"])
        pipeline_trace["stages"]["executor"] = stage4
        raw_output = str(stage4.get("raw_output") or "")
        pipeline_trace["total_tokens"] += stage4["tokens_used"]
        
        # Stage 5: Verify
        print("[PIPELINE] Stage 5: Output Verifier")
        stage5 = stage5_verify_output(enhanced_prompt, raw_output, verifier_model)
        stage5 = stage5 if isinstance(stage5, dict) else {"overall_score": 75, "status": "WARN", "flagged_issues": [], "corrections": [], "hallucination_detected": False, "send_to_improver": False}
        pipeline_trace["stages"]["verifier"] = stage5
        score = stage5.get("overall_score", 75)
        verification_status = stage5.get("status", "WARN")
        flagged_issues = stage5.get("flagged_issues", [])
        pipeline_trace["total_tokens"] += stage5.get("tokens_used", 0)
        
        # Stage 6: Improve (only if needed)
        if score < 90 and flagged_issues:
            print("[PIPELINE] Stage 6: Improver")
            stage6 = stage6_improve_output(raw_output, enhanced_prompt, flagged_issues, selected_model)
            pipeline_trace["stages"]["improver"] = stage6
            final_output = str(stage6.get("improved_output") or raw_output or "")
            pipeline_trace["total_tokens"] += stage6.get("tokens_used", 0)
            improvement_applied = True
        else:
            final_output = str(raw_output or "")
            improvement_applied = False
            pipeline_trace["stages"]["improver"] = {
                "stage": "improver",
                "status": "skipped",
                "reason": f"Score {score} >= 90 threshold"
            }
        
        pipeline_trace["status"] = "completed"
        pipeline_trace["final_output"] = final_output
        pipeline_trace["verification_status"] = verification_status
        pipeline_trace["final_score"] = score
        pipeline_trace["improvement_applied"] = improvement_applied
        
    except Exception as e:
        pipeline_trace["status"] = "error"
        pipeline_trace["error"] = str(e)
        import traceback
        pipeline_trace["traceback"] = traceback.format_exc()
    
    return pipeline_trace


# Helper function for token counting (fallback if not in ollama_client)
def count_tokens_local(text: str) -> int:
    """Estimate token count (4 chars ≈ 1 token)"""
    return max(1, len(text) // 4)
