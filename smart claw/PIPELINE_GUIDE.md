# Dolphax AI Pipeline System - Complete Build Guide

## 🎯 Overview
You now have a complete 6-stage Dolphax AI pipeline system that intelligently processes user inputs through specialized stages, cross-verifies outputs, and improves them if needed.

## 📋 System Architecture

### 6-Stage Pipeline (Sequential Execution)

```
1. PROMPT ENHANCER → 2. TASK CLASSIFIER → 3. MODEL SELECTOR → 
4. EXECUTOR → 5. OUTPUT VERIFIER → 6. IMPROVER (conditional)
```

### Stage Details

#### Stage 1: PROMPT ENHANCER
- **Input**: Raw user prompt
- **Output**: Enhanced prompt (precise, <150 words), output format
- **Logic**: Uses `qwen3:latest` to rewrite prompt, removing redundancy
- **Token Budget**: <200 tokens

#### Stage 2: TASK CLASSIFIER  
- **Input**: Enhanced prompt, output format
- **Output**: Task type, complexity level, expected output tokens
- **Task Types**: coding, math, reasoning, summarization, research, creative, classification
- **Complexity**: simple, medium, complex
- **Token Budget**: <100 tokens

#### Stage 3: MODEL SELECTOR
- **Input**: Task type, complexity, accuracy_critical flag
- **Output**: Selected model + Different verifier model
- **Selection Logic**:
  - Priority: Accuracy → Token Efficiency → Speed
  - NO voting/council: One model selected, one different model verifies
  - Uses `router.select_best_model()` function
- **Token Budget**: 0 tokens (pure logic)

#### Stage 4: EXECUTOR
- **Input**: Selected model, enhanced prompt
- **Output**: Raw output from ollama_client
- **Token Budget**: <2000 tokens

#### Stage 5: OUTPUT VERIFIER
- **Input**: Original prompt, raw output, different verifier model
- **Output**: Score (0-100), metrics, status
- **Scoring**:
  - PASS: 90-100 (excellent)
  - WARN: 70-89 (acceptable, needs improvement)
  - FAIL: 0-69 (poor)
- **Evaluation Criteria**:
  1. Correctness (0-10): Accuracy, no hallucinations
  2. Completeness (0-10): Fully addresses task
  3. Format Quality (0-10): Matches expected format
  4. Clarity (0-10): Well-structured, understandable
- **Token Budget**: <200 tokens

#### Stage 6: IMPROVER (Conditional)
- **Trigger**: Only fires if verification score < 90
- **Input**: Raw output, flagged issues, selected model
- **Output**: Improved output
- **Constraints**: One pass only, fixes ONLY flagged issues
- **Token Budget**: <2000 tokens

---

## 🔧 API Endpoints

### New Endpoint: `/api/pipeline`
```python
POST /api/pipeline
Content-Type: application/json

{
  "prompt": "user's raw input"
}
```

**Response Structure:**
```json
{
  "status": "completed",
  "stages": {
    "enhancer": { "status": "success", "enhanced_prompt": "...", "tokens_used": 45 },
    "classifier": { "status": "success", "task_type": "coding", "complexity": "medium", "tokens_used": 32 },
    "model_selector": { "status": "success", "selected_model": "codellama:latest", "verifier_model": "deepseek-r1:latest" },
    "executor": { "status": "success", "raw_output": "...", "tokens_used": 850 },
    "verifier": { "status": "success", "overall_score": 82, "status": "WARN", "flagged_issues": [...] },
    "improver": { "status": "success", "improved_output": "...", "tokens_used": 600 }
  },
  "final_output": "...",
  "verification_status": "WARN",
  "final_score": 87,
  "improvement_applied": true,
  "total_tokens": 2500,
  "token_stats": {
    "total_tokens": 2500,
    "saved_usd": 0.00,
    "session_total_tokens": 5000,
    "session_saved_usd": 0.00
  },
  "total_processing_time": 45.2
}
```

---

## 📁 Files Modified/Created

### New Files
- **`backend/pipeline.py`** - Complete orchestrator with all 6 stages
  - `stage1_enhance_prompt()` - Prompt enhancement
  - `stage2_classify_task()` - Task classification
  - `stage3_select_model()` - Model selection (uses router.select_best_model)
  - `stage4_execute_task()` - Task execution
  - `stage5_verify_output()` - Output verification (uses rater.verify_output_with_local_model)
  - `stage6_improve_output()` - Output improvement
  - `run_pipeline()` - Main orchestrator

### Modified Files
- **`backend/main.py`**
  - Added import: `from pipeline import run_pipeline`
  - Added model: `class PipelineRequest(BaseModel)`
  - Added endpoint: `@app.post("/api/pipeline")`

- **`backend/router.py`**
  - Added function: `select_best_model()` - Enhanced model selection for pipeline
  - Returns: selected_model, verifier_model (always different), reasoning, model info

- **`backend/ollama_client.py`**
  - Added function: `count_tokens_local()` - Token counting utility

- **`backend/rater.py`**
  - Added function: `verify_output_with_local_model()` - Pipeline-specific verifier
  - Scores 0-100, evaluates correctness/completeness/format/clarity

- **`frontend/dolphax.html`**
  - Added pipeline mode toggle button in topbar
  - Added `setMode()` function to switch between regular and pipeline modes
  - Added pipeline visualization UI with stage indicators
  - Added `sendPipelineQuery()` function
  - Refactored `sendMessage()` to route to pipeline or regular endpoint
  - Added real-time stage status updates
  - Shows verification score with PASS/WARN/FAIL status
  - Shows improvement notification when Stage 6 applies fixes

---

## 🚀 How to Use

### Regular Mode (Existing)
1. Type a query
2. Click send or Cmd+Enter
3. System auto-classifies and routes to best model
4. Displays model used, confidence, processing time

### Pipeline Mode (NEW)
1. Click "Pipeline" toggle button in topbar
2. Type a query  
3. Click send
4. Watch all 6 stages execute in real-time:
   - Stage 1: Enhancer (preprocessing)
   - Stage 2: Classifier (analysis)
   - Stage 3: Model Selector (selection)
   - Stage 4: Executor (generation)
   - Stage 5: Verifier (validation)
   - Stage 6: Improver (if needed)
5. See final verification score and status
6. Get improved output if score was < 90

---

## 🎯 Model Selection Priority

### Accuracy Priority (accuracy_critical=True)
- **Coding**: codellama → verified by deepseek/qwen
- **Math/Reasoning**: deepseek → verified by qwen/codellama
- **General/Creative**: qwen → verified by deepseek/codellama

### Efficiency Priority (accuracy_critical=False)
- **Coding**: codellama → verified by qwen
- **Others**: qwen → verified by deepseek/codellama

---

## 💡 Token Budgets

| Stage | Input Budget | Output Budget | Total |
|-------|-------------|---------------|-------|
| Enhancer | ∞ | <200 | <200 |
| Classifier | <200 | <100 | <100 |
| Selector | - | - | 0 |
| Executor | <300 | <2000 | <2000 |
| Verifier | <2850 | <200 | <200 |
| Improver | <3050 | <2000 | <2000 |
| **Total** | - | - | **~6500** |

---

## ⚙️ Configuration

### Ollama Models Used
```
- codellama:latest     # Code generation (accuracy: 9/10)
- deepseek-r1:latest  # Math/Reasoning (accuracy: 9/10)  
- qwen3:latest        # General/Creative (accuracy: 8/10)
```

### Environment Variables
```
OLLAMA_BASE_URL=http://localhost:11434  # Default Ollama endpoint
GEMINI_API_KEY=                          # Optional (not used in local_only mode)
OPENAI_API_KEY=                          # Optional (not used in local_only mode)
```

---

## 🧪 Testing the Pipeline

### Start Backend
```bash
cd smart\ claw/backend
uvicorn main:app --reload
```

### Test via Curl
```bash
curl -X POST http://localhost:8000/api/pipeline \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to sort a list"}'
```

### Test via Frontend
1. Navigate to http://localhost:8000 in browser
2. Click "Pipeline" mode toggle
3. Type: "Write a Python function to sort a list"
4. Watch the 6 stages execute in real-time

---

## 📊 What Makes This Pipeline Powerful

✅ **Smart Routing**: Task-specific model selection based on requirements
✅ **Cross-Verification**: Different model validates output independently
✅ **Quality Control**: Automatic improvement if verification score < 90
✅ **Token Efficiency**: Pre-engineered prompts, token budgets per stage
✅ **Zero API Cost**: Uses local Ollama models, no external API calls
✅ **Transparent**: See exactly which model validates, why, and the score
✅ **Adaptive**: Task complexity influences model selection and verification rigor
✅ **One-Pass Improvement**: Fixes only flagged issues, prevents over-editing

---

## 🔄 Pipeline Flow Example

**User Input**: "Write Python code to calculate fibonacci numbers"

```
Stage 1: ENHANCE
  - Input: "Write Python code to calculate fibonacci numbers"
  - Output: "Create a Python function that computes Fibonacci numbers"
  - Output Format: "Python code"

Stage 2: CLASSIFY
  - Task Type: "coding"
  - Complexity: "simple"
  - Accuracy Critical: true (code needs to be correct)
  - Expected Tokens: 150

Stage 3: SELECT MODEL
  - Task: coding, accuracy critical → select codellama
  - Verifier: must be different → select deepseek
  - Reasoning: "CodeLlama specializes in coding; DeepSeek will verify logic"

Stage 4: EXECUTE
  - Model: codellama:latest
  - Output: [Generated fibonacci function code]
  - Tokens: 127

Stage 5: VERIFY
  - Model: deepseek-r1:latest (different!)
  - Correctness: 9/10
  - Completeness: 8/10
  - Format: 9/10
  - Clarity: 8/10
  - Overall Score: 85/100 → WARN status
  - Issues: ["Could add type hints", "Missing docstring"]

Stage 6: IMPROVE (because score < 90)
  - Model: codellama (original executor)
  - Fixes only: type hints + docstring
  - Output: [Improved code]
  - Tokens: 145

Final Output: Python code with type hints, docstring, correct logic
Final Score: 89/100 (WARN) or improved score if verifier runs again
```

---

## 🎓 Learning from the System

This pipeline demonstrates:
1. **Orchestration**: Chaining AI stages effectively
2. **Quality Control**: Verification by independent model
3. **Efficiency**: Token budgets and prompt optimization
4. **Modularity**: Each stage is independent and reusable
5. **Transparency**: Full visibility into decision-making

---

## 📝 Notes

- Pipeline always returns complete JSON with all stage details
- Each stage's output becomes the next stage's input
- Verifier model is **always different** from executor model
- Improvement only happens if score < 90 (configurable threshold)
- All processing happens locally - no external API dependencies
- Frontend shows real-time progress through all 6 stages

---

**Build Date**: April 22, 2026  
**Status**: Production Ready ✅
