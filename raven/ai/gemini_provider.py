from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal, cast

from google import genai
from google.genai import types
import frappe

from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent, Usage
from agents.models.interface import Model, ModelProvider, ModelTracing
from agents.tool import Tool
# Removed problematic openai imports causing 'openai.resources' errors
from typing import TypedDict

# Import openai response types for agents SDK compatibility.
# Newer openai-agents versions require typed objects (not raw dicts) in ModelResponse.output.
try:
    from openai.types.responses import (
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseFunctionToolCall as OAIResponseFunctionToolCall,
    )
    _OPENAI_TYPES_AVAILABLE = True
except ImportError:
    _OPENAI_TYPES_AVAILABLE = False

if TYPE_CHECKING:
    from agents.model_settings import ModelSettings
    # ResponsePromptParam often causes issues in different versions, using Any for safety in types
    ResponsePromptParam = Any


class GeminiModel(Model):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        # Delay initialization or set properly if api key is provided
        self.client = genai.Client(api_key=api_key) if api_key else None

    async def close(self) -> None:
        pass

    def _decode_call_id(self, raw_id: str) -> tuple[str, bytes | None]:
        """Strip ||ts||<b64> suffix from a call_id and return (clean_id, thought_sig_bytes)."""
        if raw_id and "||ts||" in str(raw_id):
            parts = str(raw_id).split("||ts||", 1)
            try:
                return parts[0], base64.b64decode(parts[1])
            except Exception:
                return parts[0], None
        return raw_id, None

    def _convert_input_to_gemini(self, input_items: str | list[TResponseInputItem]) -> list[types.Content]:
        """Convert agents SDK conversation history to Gemini Contents.

        Handles two input formats:
        - Responses API format (type="function_call" / "function_call_output" / "message")
          which is what openai-agents sends on turns 2+
        - OpenAI chat format (role="assistant" with tool_calls / role="tool")
          which is used for manual conversation_history passed in turn 1
        """
        if isinstance(input_items, str):
            return [types.Content(role="user", parts=[types.Part(text=input_items)])]

        # First pass: build call_id → name map so function_call_output can include the name
        call_id_to_name: dict[str, str] = {}
        for item in input_items:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type == "function_call":
                raw_id = item.get("call_id") or item.get("id") if isinstance(item, dict) else getattr(item, "call_id", None)
                fn_name = item.get("name", "") if isinstance(item, dict) else getattr(item, "name", "")
                if raw_id:
                    clean_id, _ = self._decode_call_id(str(raw_id))
                    call_id_to_name[clean_id] = fn_name

        gemini_contents = []
        for item in input_items:
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)

            # Skip system messages — handled via system_instruction parameter
            if role == "system":
                continue

            # ── Responses API: type="function_call" (assistant tool call) ──────────
            if item_type == "function_call":
                fn_name = item.get("name", "") if isinstance(item, dict) else getattr(item, "name", "")
                fn_args = item.get("arguments", "{}") if isinstance(item, dict) else getattr(item, "arguments", "{}")
                raw_id = item.get("call_id") or item.get("id") if isinstance(item, dict) else getattr(item, "call_id", None) or getattr(item, "id", None)
                call_id, thought_sig = self._decode_call_id(str(raw_id) if raw_id else "")
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        fn_args = {}
                gemini_contents.append(types.Content(
                    role="model",
                    parts=[types.Part(
                        function_call=types.FunctionCall(id=call_id, name=fn_name, args=fn_args),
                        thought_signature=thought_sig,
                    )],
                ))
                continue

            # ── Responses API: type="function_call_output" (tool result) ─────────
            if item_type == "function_call_output":
                raw_id = item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)
                call_id, _ = self._decode_call_id(str(raw_id) if raw_id else "")
                fn_name = call_id_to_name.get(call_id, "")
                output = item.get("output") if isinstance(item, dict) else getattr(item, "output", "")
                try:
                    res_val = json.loads(output) if isinstance(output, str) else output
                except Exception:
                    res_val = {"output": output}
                if not isinstance(res_val, dict):
                    res_val = {"result": res_val}
                gemini_contents.append(types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            id=call_id,
                            name=fn_name,
                            response=res_val,
                        )
                    )],
                ))
                continue

            # ── Map role to Gemini role ───────────────────────────────────────────
            gemini_role = "model" if role == "assistant" else "user"
            parts = []

            # ── Text content (str or list of content parts) ───────────────────────
            if isinstance(content, str) and content:
                parts.append(types.Part(text=content))
            elif isinstance(content, list):
                for part in content:
                    ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                    # Accept "text" (chat format), "output_text" and "input_text" (Responses API)
                    if ptype in ("text", "output_text", "input_text"):
                        text_val = part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                        if text_val:
                            parts.append(types.Part(text=text_val))
                    elif isinstance(part, str):
                        parts.append(types.Part(text=part))

            # ── OpenAI chat format: tool_calls on assistant message ───────────────
            tool_calls = item.get("tool_calls") if isinstance(item, dict) else None
            if tool_calls:
                for tc in tool_calls:
                    ptype = tc.get("type") if isinstance(tc, dict) else getattr(tc, "type", None)
                    if ptype == "function":
                        fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                        fn_name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                        fn_args = fn.get("arguments", "{}") if isinstance(fn, dict) else getattr(fn, "arguments", "{}")
                        raw_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        call_id, thought_sig = self._decode_call_id(str(raw_id) if raw_id else "")
                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except Exception:
                                fn_args = {}
                        parts.append(types.Part(
                            function_call=types.FunctionCall(id=call_id, name=fn_name, args=fn_args),
                            thought_signature=thought_sig,
                        ))

            # ── OpenAI chat format: role="tool" ───────────────────────────────────
            if role == "tool":
                fn_name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
                raw_id = item.get("tool_call_id") if isinstance(item, dict) else getattr(item, "tool_call_id", None)
                call_id, _ = self._decode_call_id(str(raw_id) if raw_id else "")
                try:
                    res_val = json.loads(content) if isinstance(content, str) else content
                except Exception:
                    res_val = {"output": content}
                parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id,
                        name=fn_name,
                        response=res_val if isinstance(res_val, dict) else {"result": res_val},
                    )
                ))

            if parts:
                gemini_contents.append(types.Content(role=gemini_role, parts=parts))

        return gemini_contents

    def _cleanup_schema(self, schema: dict) -> dict:
        """
        Recursively remove unsupported fields from JSON schema for Gemini.
        Gemini's FunctionDeclaration is extremely strict and will fail on 'additionalProperties', 
        'title', 'description' (at property level), and other common JSON schema fields.
        """
        if not isinstance(schema, dict):
            return schema

        # Create a deep copy
        cleaned = schema.copy()

        # Remove unsupported fields from this level
        unsupported = [
            "additionalProperties", "additional_properties", "default", 
            "examples", "title", "description", "format", "pattern", 
            "minimum", "maximum", "minLength", "maxLength"
        ]
        for field in unsupported:
            cleaned.pop(field, None)

        # Handle properties - Gemini only wants type and properties at top level
        # and type + items/properties at nested levels.
        if "properties" in cleaned and isinstance(cleaned["properties"], dict):
            cleaned["properties"] = {k: self._cleanup_schema(v) for k, v in cleaned["properties"].items()}

        # Handle array items
        if "items" in cleaned and isinstance(cleaned["items"], dict):
            cleaned["items"] = self._cleanup_schema(cleaned["items"])

        # Determine type if missing but properties exist
        if "properties" in cleaned and "type" not in cleaned:
            cleaned["type"] = "object"

        return cleaned

    def _convert_tools(self, tools: list[Tool]) -> list[types.Tool]:
        """Convert agents SDK tools to Gemini native Tool objects"""
        gemini_tools = []
        function_declarations = []
        for tool in tools:
            # Only FunctionTool is easily convertible to Gemini
            if hasattr(tool, "params_json_schema"):
                # Clean up the schema for Gemini compatibility
                params = self._cleanup_schema(tool.params_json_schema or {"type": "object", "properties": {}})
                
                # Ensure type is set to object if properties exist
                if "properties" in params and "type" not in params:
                    params["type"] = "object"
                
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=params
                    )
                )
        
        if function_declarations:
            gemini_tools.append(types.Tool(function_declarations=function_declarations))
            
        return gemini_tools

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> ModelResponse:
        
        if not self.client:
            raise Exception("Gemini client not initialized. Check API Key.")

        # If input has system prompt embedded inside the list, extract it as system_instruction
        if isinstance(input, list):
            embedded_system_prompts = [item.get("content", "") for item in input if item.get("role") == "system"]
            if embedded_system_prompts:
                extracted_sys = "\n".join(embedded_system_prompts)
                system_instructions = f"{system_instructions}\n{extracted_sys}" if system_instructions else extracted_sys

        contents = self._convert_input_to_gemini(input)
        gemini_tools = self._convert_tools(tools)

        config_kwargs = {
            "temperature": model_settings.temperature,
            "top_p": model_settings.top_p,
            "candidate_count": 1,
        }
        
        if system_instructions:
            config_kwargs["system_instruction"] = system_instructions
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
            
        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
        except Exception as e:
            frappe.log_error(f"Gemini API Error: {str(e)}", "Gemini Native Provider")
            raise Exception(f"Failed to generate content via Gemini API: {str(e)}")

        output_items: list[Any] = []
        if response.candidates:
            for candidate in response.candidates:
                if not candidate.content or not hasattr(candidate.content, "parts"):
                    continue

                parts = candidate.content.parts

                # Extract text parts (exclude thought-only parts which have no actual text)
                text_parts = [p.text for p in parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False)]
                if text_parts:
                    msg_id = f"msg_{frappe.generate_hash(length=12)}"
                    joined_text = "".join(text_parts)
                    if _OPENAI_TYPES_AVAILABLE:
                        output_items.append(ResponseOutputMessage(
                            id=msg_id,
                            role="assistant",
                            status="completed",
                            type="message",
                            content=[ResponseOutputText(
                                type="output_text",
                                text=joined_text,
                                annotations=[],
                            )],
                        ))
                    else:
                        output_items.append({
                            "id": msg_id,
                            "role": "assistant",
                            "status": "completed",
                            "type": "message",
                            "content": [{"type": "output_text", "text": joined_text, "annotations": []}],
                        })

                # Collect thought_signature from any thought part as fallback.
                # On thinking models (gemini-2.0/2.5) the signature often lives on
                # the dedicated thought part, not on the function_call part itself.
                response_thought_sig = None
                for p in parts:
                    if getattr(p, "thought", False) and getattr(p, "thought_signature", None):
                        response_thought_sig = p.thought_signature
                        break

                # Extract function calls
                for part in parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call

                        # Extract and encode thought_signature into the call_id.
                        # Prefer the signature on the function_call part itself; fall back
                        # to the one found on the sibling thought part.
                        raw_id = fc.id or frappe.generate_hash(length=12)
                        thought_sig = getattr(part, "thought_signature", None) or response_thought_sig

                        encoded_id = raw_id
                        if thought_sig:
                            # base64-encode bytes so they survive as a plain string in call_id
                            if isinstance(thought_sig, bytes):
                                ts_str = base64.b64encode(thought_sig).decode("ascii")
                            else:
                                ts_str = str(thought_sig)
                            encoded_id = f"{raw_id}||ts||{ts_str}"

                        args_dict = fc.args if isinstance(fc.args, dict) else {}
                        args_json = json.dumps(args_dict)

                        if _OPENAI_TYPES_AVAILABLE:
                            output_items.append(OAIResponseFunctionToolCall(
                                id=encoded_id,
                                call_id=encoded_id,
                                type="function_call",
                                status="completed",
                                name=fc.name,
                                arguments=args_json,
                            ))
                        else:
                            output_items.append({
                                "id": encoded_id,
                                "call_id": encoded_id,
                                "type": "function_call",
                                "status": "completed",
                                "name": fc.name,
                                "arguments": args_json,
                            })

        usage_metadata = response.usage_metadata
        usage = Usage(
            input_tokens=(usage_metadata.prompt_token_count or 0) if usage_metadata else 0,
            output_tokens=(usage_metadata.candidates_token_count or 0) if usage_metadata else 0,
            total_tokens=(usage_metadata.total_token_count or 0) if usage_metadata else 0
        )

        return ModelResponse(output=output_items, usage=usage, response_id=frappe.generate_hash(length=12))

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        # Not fully implemented yet, raises NotImplementedError on usage
        raise NotImplementedError("Streaming not yet implemented for standard GeminiProvider")


class GeminiProvider(ModelProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_model(self, model_name: str | None) -> Model:
        # Default model if none specified
        return GeminiModel(model_name or "gemini-flash-latest", self.api_key)
