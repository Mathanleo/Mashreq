import json
import time
import backoff
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OPENAI_API_KEY = "openai_api_key"

class LLMClient:
    def __init__(self, config, logger):
        """
        intent_mapping = intentVsDescriptionMapping["intents"]
        group_mapping = GroupVsIntentMapping["groups"]
        """
        self.logger = logger
        self.config = config
        
        # self.allowed_intents = config["intents"]
        
        intent_mapping = config["intents"]
        group_mapping = config["groups"]

        self.all_intents = intent_mapping
        self.all_groups = group_mapping

        # Pre-build group prompt
        
        self.group_prompt = self.build_group_prompt()
        
    def build_group_prompt(self):
        """
        Build prompt for GROUP classification
        """
        group_str = "\n".join([
            f"{g['group_name']} (ID: {g['group_id']}): {g['description']}"
            for g in self.all_groups
        ])
        
        prompt = f"""
You are a banking intent GROUP classifier.

Task:
1. Identify which intent GROUP the user's utterance belongs to.
2. Match only against the list of intent groups provided below.
3. NEVER guess. If no clear match exists, return {{}}
4. Respond ONLY in valid JSON.
5. No markdown, no explanation.

Intent Groups:
{group_str}

Rules:
1. Choose the group whose description best matches the user utterance.
2. If multiple groups match, choose the closest one.
3. If nothing matches confidently, return {{}}

Respond ONLY in this JSON format:
{{
"group_name": "<group_name>",
"confidence": "<High|Medium|Low>"
}}
"""
        return prompt.strip()
    
    def build_intent_prompt(self, allowed_intents):
        intents_str = "\n".join([
            f"{i['intent_name']}: {i['description']}"
            for i in allowed_intents
        ])
        prompt = f"""
You are an intent classifier and digital banking assistant.
You MUST:
1. Match this user utterance EXACTLY to the predefined list of intents provided below.
2. If no match exists, return an empty dictionary.
3. NEVER guess or approximate an intent.

List of Intents and Descriptions:
{intents_str}

Rules:
1. If no intents score above threshold, return an empty dictionary.
2. No deviations from these rules.
3. Respond ONLY in English.
4. DO NOT include explanatory text.
5. DO NOT use markdown or code blocks.

Respond ONLY in valid JSON:
{{
"intent": "<identified_intent>",
"confidence": "<High|Medium|Low>"
}}
"""
        return prompt.strip()

        # Pre-build group prompt
        # self.group_prompt = self.build_group_prompt()

    def get_token(self):
        token_url = self.config['auth']['token_url']
        payload = {
            "grant_type": "client_credentials",
            "scope": self.config['auth']['scope'],
            "client_id": self.config['auth']['client_id'],
            "client_secret": self.config['auth']['client_secret'],
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = requests.post(token_url, headers=headers, data=payload, verify=False)  # SSL bypass for internal cert
        
        return response.json().get("access_token")

    @backoff.on_exception(backoff.expo, Exception, max_tries = 5)
    
    def call_llm(self, system_prompt, utterance):
        token = self.get_token()
        azure_endpoint = self.config['azure']['endpoint']
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "ClientId": self.config['auth']['client_id'],
             "X-USER-ID" : self.config['azure']['x_user_id'],
        }
        
        data = {
            "model": self.config['azure']['model'],
            "max_tokens": self.config['azure']['max_token'],
            "temperature": self.config['azure']['temperature'],
            "messages": [
                {"role": "user", "content": utterance},
                {"role": "system", "content": system_prompt},
            ]
        }
        
        start_time = time.time()
        response = requests.post(azure_endpoint, headers=headers, json=data, verify=False)
        elapsed = -1
        elapsed = time.time() - start_time
        elapsed = round(elapsed, 3)
        self.logger.info(f"LLM Raw Response: {response.text}")

        try:
            parsed = response.json()
            content = parsed["choices"][0]["message"]["content"]
            tokens_used = parsed["usage"]["total_tokens"]
            
            return content, tokens_used, elapsed
        except Exception:
            return "{}", 0, elapsed

    # def call_llm(self, system_prompt, user_input):
    #     url = "https://api.openai.com/v1/chat/completions"

    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {OPENAI_API_KEY}"
    #     }
        
    #     data = {
    #         "model": "gpt-4o-mini",      # choose a model you want
    #         "temperature": 0.0,
    #         "messages": [
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_input}
    #         ]
    #     }
    #     start_time = time.time()
    #     response = requests.post(url, headers=headers, json=data)
    #     elapsed = time.time() - start_time
    #     elapsed = round(elapsed, 3)

    #     parsed = response.json()

    #     content = parsed["choices"][0]["message"]["content"]
    #     tokens_used = parsed["usage"]["total_tokens"]


    #     return content, tokens_used, elapsed


    # 1. classify group
    
    def classify_group(self, utterance):
        raw, tokens, elapsed = self.call_llm(self.group_prompt, utterance)
        
        try:
            parsed = json.loads(raw)
        except:
            return {}, tokens
        
        # ensure group_name exists
        
        if not parsed or "group_name" not in parsed:
            return {}, tokens
        
        return parsed, tokens, elapsed
    
    # 2. classify intent
    
    def classify_intent(self, utterance, group_name):
        # filter intents for this group
    
        group = next((g for g in self.all_groups if g["group_name"] == group_name), None)
    
        if not group:
            return {}, 0
        
        allowed_ids = group["intents"]

        filtered_intents = [
            i for i in self.all_intents if i["intent_id"] in allowed_ids
        ]

        intent_prompt = self.build_intent_prompt(filtered_intents)
        raw, tokens, elapsed = self.call_llm(intent_prompt, utterance)

        try:
            parsed = json.loads(raw)
        except:
            return {"intent": "INVALID_JSON", "confidence": "Low"}, tokens

        return parsed, tokens, elapsed
    
    # Two step classification
    
    def classify(self, utterance):
        # STEP 1 - GROUP
        
        group_result, g_tokens, group_time = self.classify_group(utterance)
        
        if not group_result:
            return {
                "group": None,
                "intent": None,
                "confidence": "Low",
                "intent_time_sec": 0,
                "group_time_sec": group_time,
                "total_tokens_used": g_tokens,
            }
        
        group_name = group_result["group_name"]
        
        # STEP 2 - INTENT
        
        intent_result, i_tokens, intent_time = self.classify_intent(utterance, group_name)
        
        return {
            "group": group_result,
            "intent": intent_result,
            "group_time_sec": group_time,
            "intent_time_sec": intent_time,
            "total_tokens_used": g_tokens + i_tokens,
        }
        
        # Example Output:
        # {
        #     "group": {
        #         "group_name": "Account Information",
        #         "confidence": "High"
        #     },
        #     "intent": {
        #         "intent": "Get Balance",
        #         "confidence": "High"
        #     },
        #     "total_tokens_used": 152
        # }