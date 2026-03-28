This document outlines the architectural decisions and technical constraints discovered during the integration of the **Google Gemini Native Provider** (using `google-genai` SDK v1.x) into the **Raven AI Framework** (built on the Frappe framework).

## 🏙️ Ecosystem: Frappe & RavenChat AI

### Frappe Framework Context
Raven is a Frappe application. The AI integration relies on standard Frappe patterns:
- **Configuration**: API keys and provider-specific settings are stored in the **Raven Settings** DocType.
- **Logging**: All Gemini API errors and schema mismatches are logged to the **Error Log** DocType (`/app/error-log`) with the `Gemini Native Provider` identifier.
- **Hooks**: Raven uses DocType class overrides and background tasks (via Frappe Redis) to handle AI messaging asynchronously.

### Raven AI Feature Architecture
The Gemini integration is plugged into Raven's feature discovery system:
- **Model Discovery**: `raven/api/ai_features.py` defines the `get_gemini_available_models` method, which dynamically polls the Google API to list models supporting `generateContent`.
- **Provider Interface**: The `GeminiProvider` class in `raven/ai/gemini_provider.py` implements the standard `ModelProvider` interface required by Raven.
- **Runtime**: AI responses are processed through the `agents` SDK, then converted back to Raven/Frappe Message DocTypes for display in the chat UI.

## 🧱 Modular Development & Upgradability

This implementation is designed to be **modular** to facilitate easier merging with the official Raven repository. 

### Guidelines for Future Development:
- **Isolated Provider Logic**: Keep all Gemini-specific API handling, schema cleanup, and signature propagation within `raven/ai/gemini_provider.py`. Avoid spreading provider-specific logic into the core message handling loop.
- **Generic Hooks**: When modifying `raven/api/ai_features.py`, use generic patterns that could support other future providers (e.g., Anthropic, Groq) to ensure the core API surface remains clean.
- **SDK Standard Compliance**: Always adhere strictly to the `agents` SDK interfaces (`Model`, `ModelProvider`). By following the official SDK patterns, we ensure that as Raven's core messaging engine evolves, the Gemini provider remains a "drop-in" component.
- **Defensive Imports**: Continue using the pattern of local/try-except imports for heavy dependencies like `openai` or `google-genai`. This prevents the entire Raven app from crashing if a specific library is missing or outdated in a user's environment.

## 🏗️ Architecture Overview

The integration replaces the previous "OpenAI-compatible" approach, which was prone to schema validation errors and lacked support for Gemini-specific features like `thought_signature`.

### Key Components
- **Provider**: `raven/ai/gemini_provider.py`
- **SDK**: `google-genai` (v1.x)
- **Integration Layer**: `raven/api/ai_features.py` (Model Discovery)

---

## 💡 Technical Learnings & Constraints

### 1. Strict JSON Schema Validation
Gemini's `FunctionDeclaration` is significantly stricter than OpenAI's.
- **Unsupported Fields**: Recursively strip `additionalProperties`, `title`, `description` (at the property level), `default`, `examples`, `format`, `pattern`, `minimum`, `maximum`.
- **Object Types**: If a schema has `properties`, it **must** have `"type": "object"`. Failure to provide this explicitly often causes `400 INVALID_ARGUMENT`.

### 2. Multi-Turn Tool Use & "Max Turns Exceeded"
A persistent "Max turns (5) exceeded" error was resolved by implementing **Thought Signature Propagation**.
- **The Issue**: Models like Gemini 2.0 generate internal reasoning (`thought`). If a tool call is passed back in history without its original `thought_signature`, the model enters an infinite loop.
- **The Fix**: We encode the `thought_signature` into the `call_id` returned to the SDK. During the next turn, we decode it and pass it back in the `types.Part(function_call=..., thought_signature=...)` object.

### 3. Role Mapping
Gemini requires strict role transitions:
- **Assistant Message**: Must use `role: "model"`.
- **Tool Result**: Must use `role: "user"`.
- **System Prompt**: Should be passed via the `system_instruction` parameter in `GenerateContentConfig`, NOT as a message in the conversation history.

### 4. Agents SDK Pydantic Requirements
The `agents` SDK (v0.10+) enforces strict validation on `ModelResponse`:
- **Unique IDs**: Every message and tool call must have a unique string `id`.
- **Output Types**: 
  - Text messages must have `type: "message"` and content items with `type: "output_text"`.
  - Tool calls must have `type: "function_call"` (NOT "function").
- **Usage**: Token counts (`input_tokens`, `output_tokens`) must be integers. Use `(count or 0)` to avoid `None` errors.

---

## 🛠️ Maintenance & Troubleshooting

### Bench Restart
Always run `bench restart` after modifying `gemini_provider.py` or `ai_features.py`. Frappe/Python's module cache can cause old logic (or missing imports) to persist.

### Import Resilience
The codebase uses defensive imports for `openai` and other libraries to ensure high availability even if specific SDK sub-modules are missing in certain environments.

### Debugging API Errors
Check the **Frappe Error Log** (`/app/error-log`) for "Gemini Native Provider" entries. Most failures are `400 INVALID_ARGUMENT` due to schema mismatches in newly added tools.
