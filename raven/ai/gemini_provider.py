from __future__ import annotations

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

    def _convert_input_to_gemini(self, input_items: str | list[TResponseInputItem]) -> list[types.Content]:
        """Convert OpenAI-style history to Gemini Contents"""
        if isinstance(input_items, str):
            return [types.Content(role="user", parts=[types.Part(text=input_items)])]

        gemini_contents = []
        for item in input_items:
            role = item.get("role")
            content = item.get("content")
            
            # Skip system role (handled via system_instruction parameter)
            if role == "system":
                continue

            # Map OpenAI/Raven roles to Gemini roles
            # Gemini: 'user' or 'model' (for assistant)
            gemini_role = "model" if role == "assistant" else "user"
            parts = []

            # 1. Handle Text Content
            if isinstance(content, str) and content:
                parts.append(types.Part(text=content))
            elif isinstance(content, list):
                for part in content:
                    ptype = getattr(part, "type", part.get("type", None) if isinstance(part, dict) else None)
                    if ptype == "text":
                        text_val = getattr(part, "text", part.get("text", "") if isinstance(part, dict) else "")
                        if text_val:
                            parts.append(types.Part(text=text_val))
                    elif isinstance(part, str):
                        parts.append(types.Part(text=part))

            # 2. Handle Tool Calls (Assistant Message)
            tool_calls = item.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    ptype = getattr(tc, "type", tc.get("type") if isinstance(tc, dict) else None)
                    if ptype == "function":
                        fn = getattr(tc, "function", tc.get("function") if isinstance(tc, dict) else {})
                        fn_name = getattr(fn, "name", fn.get("name") if isinstance(fn, dict) else "")
                        fn_args = getattr(fn, "arguments", fn.get("arguments", "{}") if isinstance(fn, dict) else "{}")
                        
                        # Handle encoded thought_signature in call_id
                        raw_id = getattr(tc, "id", tc.get("id"))
                        call_id = raw_id
                        thought_sig = None
                        if raw_id and "||ts||" in str(raw_id):
                            parts_id = str(raw_id).split("||ts||")
                            call_id = parts_id[0]
                            thought_sig = parts_id[1]
                        
                        # Gemini expects args as dict
                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except:
                                fn_args = {}
                                
                        parts.append(types.Part(
                            function_call=types.FunctionCall(
                                id=call_id,
                                name=fn_name,
                                args=fn_args
                            ),
                            thought_signature=thought_sig
                        ))

            # 3. Handle Tool Responses (Tool Message)
            if role == "tool":
                fn_name = item.get("name")
                # Handle encoded thought_signature in tool_call_id
                raw_id = item.get("tool_call_id")
                call_id = raw_id
                if raw_id and "||ts||" in str(raw_id):
                    parts_id = str(raw_id).split("||ts||")
                    call_id = parts_id[0]
                
                # Gemini expects function_response to be a dict
                try:
                    res_val = json.loads(content) if isinstance(content, str) else content
                except:
                    res_val = {"output": content}
                
                parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id,
                        name=fn_name,
                        response=res_val if isinstance(res_val, dict) else {"result": res_val}
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

        # DEBUG: log raw Gemini response
        try:
            debug_lines = [
                f"model={self.model_name}",
                f"candidates={len(response.candidates) if response.candidates else 0}",
                f"prompt_feedback={getattr(response, 'prompt_feedback', None)}",
                f"usage={response.usage_metadata}",
            ]
            if response.candidates:
                for ci, cand in enumerate(response.candidates):
                    debug_lines.append(f"candidate[{ci}].finish_reason={getattr(cand, 'finish_reason', None)}")
                    if cand.content and hasattr(cand.content, "parts"):
                        for pi, p in enumerate(cand.content.parts):
                            debug_lines.append(
                                f"  part[{pi}]: text={repr(p.text)[:120] if hasattr(p, 'text') else 'N/A'}"
                                f" | thought={getattr(p, 'thought', None)}"
                                f" | has_fc={bool(getattr(p, 'function_call', None))}"
                                f" | has_ts={bool(getattr(p, 'thought_signature', None))}"
                            )
            frappe.log_error("\n".join(debug_lines), "Gemini Debug")
        except Exception as _dbg_e:
            frappe.log_error(f"Debug logging failed: {_dbg_e}", "Gemini Debug")

        output_items: list[dict[str, Any]] = []
        if response.candidates:
            for candidate in response.candidates:
                if not candidate.content or not hasattr(candidate.content, "parts"):
                    continue

                parts = candidate.content.parts

                # Extract text parts
                text_parts = [p.text for p in parts if hasattr(p, "text") and p.text]
                if text_parts:
                    msg_id = f"msg_{frappe.generate_hash(length=12)}"
                    output_items.append({
                        "id": msg_id,
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                        "content": [{
                            "type": "output_text", 
                            "text": "".join(text_parts),
                            "annotations": []
                        }]
                    })
                
                # Extract function calls
                for part in parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        
                        # Extract and encode thought_signature into the call_id
                        raw_id = fc.id or frappe.generate_hash(length=12)
                        thought_sig = getattr(part, "thought_signature", None)
                        
                        encoded_id = raw_id
                        if thought_sig:
                            encoded_id = f"{raw_id}||ts||{thought_sig}"
                        
                        args_dict = fc.args if isinstance(fc.args, dict) else {}
                        args_json = json.dumps(args_dict)
                        
                        output_items.append({
                            "id": encoded_id,
                            "call_id": encoded_id,
                            "type": "function_call",
                            "status": "completed",
                            "name": fc.name,
                            "arguments": args_json,
                            "function": {
                                "name": fc.name,
                                "arguments": args_json
                            }
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
