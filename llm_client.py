import requests
import json
import time
import backoff
import urllib3
import html

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OPENAI_API_KEY = "openai_api_key"

class LLMClient:
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config

        self.intent_mapping = config["intentVsDescriptionMapping"]["intents"]
        self.group_mapping = config["GroupVsIntentMapping"]["groups"]

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    def decode_html(self, text):
        return html.unescape(text.strip()) if isinstance(text, str) else text

    def map_group_name_to_id(self, group_name):
        if not group_name:
            return None
        norm = self.decode_html(group_name).lower()
        for g in self.group_mapping:
            if g["group_name"].lower() == norm:
                return int(g["group_id"])
        return None

    # --------------------------------------------------
    # Prompt Builders
    # --------------------------------------------------
    def build_group_prompt(self, utterance):
        lines = ["Groups:"]
        for g in self.group_mapping:
            lines.append(f'{g["group_id"]}) {g["group_name"]} — {g["description"]}')

        return f"""
You are a banking intent router. Pick the most relevant groups for this user utterance "{utterance}".

{chr(10).join(lines)}

Output JSON (one of the following formats):
1) Single:
{{"group_name":"Security & Authentication","group_id":"4","confidence":0.92}}
2) Multiple:
{{"Groups":[
  {{"group_name":"Security & Authentication","group_id":"4","confidence":0.92}},
  {{"group_name":"Cards & Controls","group_id":"3","confidence":0.88}}
]}}
""".strip()

    def build_intent_prompt(self, utterance, intent_text):
        return f"""
You are an intent classifier and digital banking assistant.
You MUST:
1. Match this user utterance " {utterance} " EXACTLY to the predefined list of intents provided below.
2. If no match exists, return an empty array.
3. NEVER guess or approximate an intent.

List of Intents and Descriptions:
{intent_text}

Rules:
1. Response MUST be in valid JSON format with a list of intent names along with scores.
2. Scores MUST be in the range of 0–1, with exactly two decimal places.
3. ONLY include intents with scores >= 0.6.
4. Intents MUST be sorted strictly descending by score.
5. Multiple intents may share identical scores if equally relevant.
6. If no intents score above threshold, return an empty array.
7. Each intent object MUST use exact keys "Intent" and "Score".
8. No deviations from these rules are allowed.
9. Respond ONLY in English.
10. DO NOT include explanatory text outside the JSON structure.
11. DO NOT use markdown formatting or code blocks.

Return format: {{"Intents": [{{"Intent": "<name>", "Score": <0.00–1.00>}}]}}
If no match: {{"Intents": []}}

Now return ONLY the JSON result.
""".strip()

    # --------------------------------------------------
    # LLM Call
    # --------------------------------------------------
    def get_token(self):
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.config['auth']['client_id'],
            "client_secret": self.config['auth']['client_secret'],
            "scope": self.config['auth']['scope']
        }
        res = requests.post(
            self.config['auth']['token_url'],
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data=payload,
            verify=False
        )
        return res.json().get("access_token")

    def call_llm(self, prompt):
        url = "https://api.openai.com/v1/chat/completions"
        print("The URL being called is ", url)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        data = {
            "model": "gpt-4.1-2025-04-14",  # updated model
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": prompt}
            ]
        }

        start_time = time.time()
        print("The prompt being sent to LLM is ")

        response = requests.post(url, headers=headers, json=data)
        print("The response from LLM is ", response.text)

        elapsed = round(time.time() - start_time, 3)

        parsed = response.json()

        content = parsed["choices"][0]["message"]["content"]

        # ✅ Token breakdown
        input_tokens = parsed["usage"]["prompt_tokens"]
        output_tokens = parsed["usage"]["completion_tokens"]
        total_tokens = parsed["usage"]["total_tokens"]

        self.logger.debug(f"LLM Prompt:\n{prompt}")
        self.logger.debug(f"LLM Raw Response:\n{content}")
        self.logger.debug(
            f"Input tokens: {input_tokens}, "
            f"Output tokens: {output_tokens}, "
            f"Total tokens: {total_tokens}, "
            f"Time: {elapsed}s"
        )

        print("The content is ", content)

        return content, input_tokens, output_tokens, total_tokens, elapsed


    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def call_llm1(self, prompt):
        token = self.get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-USER-ID": self.config['azure']['x_user_id'],
            "ClientId": self.config['auth']['client_id']
        }
        body = {
            "model": self.config['azure']['model'],
            "temperature": self.config['azure']['temperature'],
            "max_tokens": self.config['azure']['max_token'],
            "messages": [{"role": "user", "content": prompt}]
        }
        start = time.time()
        res = requests.post(self.config['azure']['endpoint'], headers=headers, json=body, verify=False)
        elapsed = round(time.time() - start, 3)

        content = res.json()["choices"][0]["message"]["content"]
        tokens = res.json()["usage"]["total_tokens"]
        return content, tokens, elapsed

    # --------------------------------------------------
    # Router Logic (JS parity)
    # --------------------------------------------------
    def process_router_output(self, router_out, top_k=2, min_conf=0.7, tie_delta=0.05):
        try:
            parsed = json.loads(router_out)
        except:
            return ""

        candidates = parsed.get("Groups", [parsed]) if isinstance(parsed, dict) else []

        norm = []
        for c in candidates:
            gid = c.get("group_id")
            gname = self.decode_html(c.get("group_name", ""))
            conf = float(c.get("confidence", 0))
            gid = int(gid) if gid else self.map_group_name_to_id(gname)
            if gid is not None:
                norm.append({"group_id": gid, "group_name": gname, "confidence": conf})

        if not norm:
            return ""

        norm.sort(key=lambda x: x["confidence"], reverse=True)

        selected = []
        top1 = norm[0]["confidence"]
        for c in norm:
            if len(selected) >= top_k:
                break
            if c["confidence"] >= min_conf or (top1 - c["confidence"]) <= tie_delta:
                selected.append(c)
        if not selected:
            selected.append(norm[0])

        seen, lines, idx = set(), [], 1
        for sg in selected:
            intent_ids = next(g["intents"] for g in self.group_mapping if g["group_id"] == sg["group_id"])
            intents = [i for i in self.intent_mapping if i["intent_id"] in intent_ids]
            for intent in intents:
                key = intent["intent_name"].lower()
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{idx}. {intent['intent_name']} : {intent['description']}")
                idx += 1
        print("The lines are ",lines)
        return "\n".join(lines)

    # --------------------------------------------------
    # Full Classification
    # --------------------------------------------------
    def classify(self, utterance):
        group_prompt = self.build_group_prompt(utterance)
        print("the group prompt is ",group_prompt)
        print("Calling LLM for group classification...")
        g_raw, g_input_tokens, g_output_tokens, g_total_tokens, g_time = self.call_llm(group_prompt)

        intent_text = self.process_router_output(g_raw)
        if not intent_text:
            return {"group": None, "intent": None, "total_tokens_used": g_total_tokens, "group_input_tokens": g_input_tokens, "group_output_tokens": g_output_tokens, "group_time_sec": g_time, "intent_time_sec": 0}
        intent_prompt = self.build_intent_prompt(utterance, intent_text)
        print("the intent prompt is ",intent_prompt)
        i_raw, i_input_tokens, i_output_tokens, i_total_tokens, i_time = self.call_llm(intent_prompt)
        a = {
            "group": g_raw,
            "intent": json.loads(i_raw),
            "total_tokens_used": g_total_tokens + i_total_tokens,
            # "group_input_tokens": g_input_tokens,
            # "group_output_tokens": g_output_tokens,
            # "intent_input_tokens": i_input_tokens,
            # "intent_output_tokens": i_output_tokens,
            "total_input_tokens": g_input_tokens + i_input_tokens,
            "total_output_tokens": g_output_tokens + i_output_tokens,
            "group_time_sec": g_time,
            "intent_time_sec": i_time
        }
        print("The final classification output is ",a)
        return a