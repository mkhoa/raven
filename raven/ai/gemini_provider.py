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
from openai.types.responses import ResponseOutputMessage, ResponseOutputText, ResponseFunctionToolCall

if TYPE_CHECKING:
    from agents.model_settings import ModelSettings
    from openai.types.responses.response_prompt_param import ResponsePromptParam


class GeminiModel(Model):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        # Delay initialization or set properly if api key is provided
        self.client = genai.Client(api_key=api_key) if api_key else None

    async def close(self) -> None:
        pass

    def _convert_input_to_gemini(self, input_items: str | list[TResponseInputItem]) -> list[types.Content]:
        if isinstance(input_items, str):
            return [types.Content(role="user", parts=[types.Part.from_text(text=input_items)])]

        gemini_contents = []
        for item in input_items:
            role = item.get("role")
            content = item.get("content")
            
            # Map roles exactly as Gemini expects: 'user' or 'model' (for assistant)
            # system role is not a valid conversational turn role, it is handled via GenerateContentConfig
            if role == "system":
                # System instructions should not be in the messages array for Gemini
                # the integration layer usually passes it to system_instructions parameter
                continue

            gemini_role = "user" if role in ["user", "tool"] else "model"
            
            parts = []
            if isinstance(content, str) and content:
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for part in content:
                    if getattr(part, "type", part.get("type")) == "text":
                        # Support objects or dicts
                        text_val = getattr(part, "text", part.get("text", ""))
                        if text_val:
                            parts.append(types.Part.from_text(text=text_val))

            # Handle tool calls in assistant messages
            tool_calls = item.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    if getattr(tc, "type", tc.get("type")) == "function":
                        func = getattr(tc, "function", tc.get("function"))
                        func_name = getattr(func, "name", func.get("name"))
                        func_args = getattr(func, "arguments", func.get("arguments", "{}"))
                        
                        try:
                            parsed_args = json.loads(func_args) if isinstance(func_args, str) else func_args
                        except json.JSONDecodeError:
                            parsed_args = {}
                            
                        parts.append(types.Part.from_function_call(
                            name=func_name,
                            args=parsed_args
                        ))

            # Handle tool responses
            if role == "tool":
                # ensure response is a valid struct
                tool_res = content
                try:
                    if isinstance(tool_res, str):
                        tool_res_dict = json.loads(tool_res)
                    else:
                        tool_res_dict = tool_res
                        
                    if not isinstance(tool_res_dict, dict):
                        tool_res_dict = {"result": tool_res}
                except (json.JSONDecodeError, TypeError):
                    tool_res_dict = {"result": str(tool_res)}

                parts.append(types.Part.from_function_response(
                    name=item.get("name", "unknown"), 
                    response=tool_res_dict
                ))

            if parts:
                gemini_contents.append(types.Content(role=gemini_role, parts=parts))

        return gemini_contents

    def _convert_tools_to_gemini(self, tools: list[Tool]) -> list[types.Tool]:
        gemini_tools = []
        function_declarations = []
        for tool in tools:
            # Only FunctionTool is easily convertible to Gemini
            if hasattr(tool, "params_json_schema"):
                # Make sure parameters is an object type
                params = tool.params_json_schema or {"type": "object", "properties": {}}
                if "type" not in params:
                    params["type"] = "object"
                if "properties" not in params:
                    params["properties"] = {}
                
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
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
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
        gemini_tools = self._convert_tools_to_gemini(tools)

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

        output_items = []
        if response.candidates:
            for candidate in response.candidates:
                if not candidate.content or not getattr(candidate.content, "parts", None):
                    continue
                    
                parts = candidate.content.parts
                
                # Extract text parts
                text_parts = [p.text for p in parts if getattr(p, "text", None)]
                if text_parts:
                    output_items.append(ResponseOutputMessage(
                        role="assistant",
                        content=[ResponseOutputText(type="text", text=" ".join(text_parts))]
                    ))
                
                # Extract function calls
                for part in parts:
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        fc_name = fc.name
                        fc_args = fc.args if fc.args else {}
                        args_json = json.dumps(fc_args) if isinstance(fc_args, dict) else str(fc_args)
                        
                        output_items.append(ResponseFunctionToolCall(
                            id=f"call_{fc_name}_{hash(args_json)}", # Semi-unique ID acceptable for agents parser
                            type="function",
                            function={
                                "name": fc_name,
                                "arguments": args_json
                            }
                        ))

        usage_metadata = response.usage_metadata
        usage = Usage(
            prompt_tokens=usage_metadata.prompt_token_count if usage_metadata else 0,
            completion_tokens=usage_metadata.candidates_token_count if usage_metadata else 0,
            total_tokens=usage_metadata.total_token_count if usage_metadata else 0
        )

        return ModelResponse(output=output_items, usage=usage, response_id=None)

    def stream_response(self, *args, **kwargs) -> AsyncIterator[TResponseStreamEvent]:
        # Not fully implemented yet, raises NotImplementedError on usage
        raise NotImplementedError("Streaming not yet implemented for standard GeminiProvider")


class GeminiProvider(ModelProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_model(self, model_name: str | None) -> Model:
        # Default model if none specified
        return GeminiModel(model_name or "gemini-2.5-flash", self.api_key)
