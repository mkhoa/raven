import frappe




@frappe.whitelist(methods=["GET"])
def get_instruction_preview(instruction: str):
	"""
	Function to get the rendered instructions for the bot
	"""
	frappe.has_permission(doctype="Raven Bot", ptype="write", throw=True)
	from raven.ai.handler import get_variables_for_instructions

	instructions = frappe.render_template(instruction, get_variables_for_instructions())
	return instructions


@frappe.whitelist()
def get_saved_prompts(bot: str = None):
	"""
	API to get the saved prompt for a user/bot/global
	"""
	or_filters = [["is_global", "=", 1], ["owner", "=", frappe.session.user]]

	prompts = frappe.get_list(
		"Raven Bot AI Prompt", or_filters=or_filters, fields=["name", "prompt", "is_global", "raven_bot"]
	)

	# Order by ones with the given bot
	prompts = sorted(prompts, key=lambda x: x.get("raven_bot") == bot, reverse=True)

	return prompts


@frappe.whitelist()
def get_open_ai_version():
	"""
	API to get the version of the OpenAI Python client
	"""
	frappe.has_permission(doctype="Raven Bot", ptype="read", throw=True)
	return openai.__version__


@frappe.whitelist()
def get_openai_available_models():
	"""
	API to get the available OpenAI models for assistants
	"""
	frappe.has_permission(doctype="Raven Bot", ptype="read", throw=True)
	from raven.ai.openai_client import get_openai_models

	models = get_openai_models()

	valid_prefixes = ["gpt-5", "gpt-4", "gpt-3.5", "o1", "o3-mini"]

	# Model should not contain these words
	invalid_models = ["realtime", "transcribe", "search", "audio"]

	compatible_models = []

	for model in models:
		if any(model.id.startswith(prefix) for prefix in valid_prefixes):
			if not any(word in model.id for word in invalid_models):
				compatible_models.append(model.id)

	return compatible_models


@frappe.whitelist()
def test_llm_configuration(
	provider: str = "OpenAI", api_url: str = None, local_llm_provider: str = None
):
	"""
	Test LLM configuration (OpenAI or Local LLM)
	"""
	frappe.has_permission(doctype="Raven Settings", ptype="write", throw=True)

	try:
		if provider == "Local LLM" and api_url:
			# Test local LLM endpoint
			import requests

			# Check if it's OpenAI Compatible and get API key
			api_key = None
			if local_llm_provider == "OpenAI Compatible":
				settings = frappe.get_single("Raven Settings")
				api_key = settings.get_password("openai_compatible_api_key")
				if not api_key:
					return {
						"success": False,
						"message": "OpenAI Compatible API Key is required",
					}

			if api_key:
				import openai
				# Use OpenAI client for OpenAI Compatible services
				client = openai.OpenAI(api_key=api_key, base_url=api_url)
				models = client.models.list()
				return {
					"success": True,
					"message": f"Successfully connected to OpenAI Compatible service at {api_url}",
					"models": [{"id": m.id} for m in models.data],
				}
			else:
				# For other local LLM providers, use direct HTTP request
				response = requests.get(f"{api_url}/models", timeout=5)
				if response.status_code == 200:
					models = response.json()
					return {
						"success": True,
						"message": f"Successfully connected to {api_url}",
						"models": models.get("data", []),
					}
				else:
					return {
						"success": False,
						"message": f"Failed to connect to {api_url}. Status: {response.status_code}",
					}

		elif provider == "OpenAI":
			# Test OpenAI configuration
			from raven.ai.openai_client import get_open_ai_client

			client = get_open_ai_client()
			# Try to list models
			models = client.models.list()
			return {
				"success": True,
				"message": "Successfully connected to OpenAI",
				"models": [{"id": m.id} for m in models.data[:5]],  # Return first 5 models
			}

	except Exception as e:
		return {"success": False, "message": f"Connection failed: {str(e)}"}


@frappe.whitelist()
def get_gemini_available_models():
	"""
	API to get the available Gemini models
	"""
	frappe.has_permission(doctype="Raven Bot", ptype="read", throw=True)
	
	try:
		settings = frappe.get_single("Raven Settings")
		api_key = settings.get_password("gemini_api_key")
		
		if not api_key:
			# If no key in settings, return a helpful hint + standard fallbacks
			return ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-pro-latest", "(Enter Gemini API Key in Raven Settings to see more)"]
			
		from google import genai
		client = genai.Client(api_key=api_key)
		
		# Simplify call - just use default list()
		# Use a generator approach to be safe with large lists
		models_iterator = client.models.list()
		
		compatible_models = []
		for m in models_iterator:
			# Check supported actions for generateContent
			actions = getattr(m, "supported_actions", [])
			if "generateContent" in actions:
				name = m.name.replace("models/", "")
				# Skip purely internal or non-text models
				if not any(x in name for x in ["vision", "aqa", "embedding"]):
					compatible_models.append(name)
				
		return compatible_models or ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-pro-latest"]
	except Exception as e:
		import traceback
		frappe.log_error(f"Error fetching Gemini models: {str(e)}\n{traceback.format_exc()}", "Raven AI")
		return ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-pro-latest"]


@frappe.whitelist()
def test_gemini_configuration():
	"""
	Test Gemini configuration
	"""
	frappe.has_permission(doctype="Raven Settings", ptype="write", throw=True)
	
	try:
		settings = frappe.get_single("Raven Settings")
		api_key = settings.get_password("gemini_api_key")
		
		if not api_key:
			return {"success": False, "message": "Gemini API Key is missing"}
			
		from google import genai
		client = genai.Client(api_key=api_key)
		
		# Try a minimal list models call to verify the key
		models = client.models.list()
		# Just check if we can get at least one model
		first_model = next(iter(models), None)
		
		return {
			"success": True, 
			"message": "Successfully connected to Gemini API",
			"models": [{"id": first_model.name.replace("models/", "")}] if first_model else []
		}
	except Exception as e:
		return {"success": False, "message": f"Gemini connection failed: {str(e)}"}
