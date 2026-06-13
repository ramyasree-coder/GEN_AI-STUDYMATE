import os
import json
import hashlib
import requests
import numpy as np
from typing import List, Optional
from requests import HTTPError


class GeminiClient:
    """Wrapper for Google Generative Language API / Gemini-compatible endpoints.

    This client prefers calling the model-specific REST endpoints (recommended):
      POST {api_url}/v1/models/{model}:embed
      POST {api_url}/v1/models/{model}:generate

    If the `google.generativeai` SDK is installed, the client will try to use it
    first (best-effort). If no API key is provided, a deterministic local
    fallback is used for embeddings and generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        embedding_model: str = "models/gemini-embedding-2",
        generate_model: str = "models/gemini-2.5-flash",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        # default Google Generative Language API host
        self.api_url = api_url or os.environ.get("GEMINI_API_URL") or "https://generativelanguage.googleapis.com"
        self.embedding_model = embedding_model
        self.generate_model = generate_model

        def _norm(m: str) -> str:
            if m.startswith("models/"):
                return m[len("models/"):]
            return m

        # normalize models so building URLs is consistent
        # Ordered list of generate model candidates for fallback (short names)
        self.generate_model_candidates = [
            _norm(self.generate_model),
            "gemini-2.5-flash-lite",
            "gemini-2.1",
        ]
        self._embedding_model_short = _norm(self.embedding_model)
        self._generate_model_short = _norm(self.generate_model)

        # Prefer the official SDKs if available. Try `google.genai` first (new
        # SDK), then the older `google.generativeai`. If a SDK is present we
        # will prefer it and ignore `GEMINI_API_URL` since the SDK manages
        # endpoints/auth internally.
        self._genai = None
        self._genai_name = None
        try:
            import google.genai as _genai_new  # type: ignore

            self._genai = _genai_new
            self._genai_name = "genai"
        except Exception:
            try:
                import google.generativeai as _genai_old  # type: ignore

                self._genai = _genai_old
                self._genai_name = "generativeai"
            except Exception:
                self._genai = None
                self._genai_name = None
            # SDK client will be created lazily when needed
            self._genai_client = None

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post_with_key_fallback(self, url: str, payload: dict, timeout: int = 30):
        """POST helper: try Authorization header first, then retry with ?key=API_KEY if auth fails.

        Returns the requests.Response object or raises for non-200 statuses after retries.
        """
        headers = self._headers()
        resp = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 401 or resp.status_code == 403:
                # try query param API key fallback
                if self.api_key:
                    sep = '&' if '?' in url else '?'
                    url_k = f"{url}{sep}key={self.api_key}"
                    resp = requests.post(url_k, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)
            return resp
        except Exception:
            # re-raise so callers can handle
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # If SDK available, try it first (best-effort and may vary by SDK version)
        if self._genai is not None and self.api_key:
            try:
                # Use google.genai Client when available
                if self._genai_name == "genai":
                    try:
                        self._genai_client = self._genai.Client(api_key=self.api_key)
                    except Exception:
                        self._genai_client = None
                    if self._genai_client is not None:
                        try:
                            resp = self._genai_client.models.embed_content(model=self._embedding_model_short, contents=texts)
                            data = getattr(resp, "model_dump", None) and resp.model_dump() or getattr(resp, "dict", lambda: {} )()
                            if isinstance(data, dict) and "embeddings" in data:
                                out = []
                                for e in data.get("embeddings", []):
                                    vals = e.get("values") or e.get("embedding")
                                    if vals is not None:
                                        out.append(list(vals))
                                if out:
                                    return out
                        except Exception:
                            pass

                # Fallback to older `google.generativeai` SDK
                if self._genai_name == "generativeai":
                    try:
                        self._genai.configure(api_key=self.api_key)
                    except Exception:
                        pass
                    if hasattr(self._genai, "embeddings") and hasattr(self._genai.embeddings, "create"):
                        resp = self._genai.embeddings.create(model=self.embedding_model, input=texts)
                        data = getattr(resp, "data", None) or resp
                        out = []
                        for item in data:
                            emb = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
                            if emb is not None:
                                out.append(list(emb))
                        if out:
                            return out
            except Exception:
                # fall through to direct HTTP call
                pass

        # HTTP REST call to model-specific embed endpoint
        if self.api_key and self.api_url:
            url = f"{self.api_url.rstrip('/')}/v1/models/{self._embedding_model_short}:embed"
            payload = {"input": texts} if len(texts) != 1 else {"input": texts[0]}
            resp = self._post_with_key_fallback(url, payload, timeout=30)
            if resp is None:
                raise RuntimeError("No response from embeddings endpoint")
            if resp.status_code == 404:
                return [self._deterministic_embedding(t) for t in texts]
            resp.raise_for_status()
            data = resp.json()
            # Support multiple response formats used by different SDKs/versions
            if isinstance(data, dict):
                if "embeddings" in data:
                    return [e.get("embedding") for e in data.get("embeddings", [])]
                if "data" in data:
                    return [item.get("embedding") for item in data.get("data", [])]
                if "outputs" in data:
                    # some responses nest embeddings under outputs
                    out = []
                    for o in data.get("outputs", []):
                        if isinstance(o, dict) and "embedding" in o:
                            out.append(o.get("embedding"))
                    if out:
                        return out

            # If parsing failed, raise to surface unexpected schema
            raise RuntimeError(f"Unexpected embeddings response: {data}")

        # Fallback deterministic embeddings (for offline testing)
        return [self._deterministic_embedding(t) for t in texts]

    def generate_text(self, prompt: str, max_tokens: int = 256) -> str:
        # Try candidates in order with fallback on quota/errors
        last_exc = None
        for model_short in getattr(self, "generate_model_candidates", [self._generate_model_short]):
            try:
                return self._generate_for_model(model_short, prompt, max_tokens)
            except Exception as e:
                last_exc = e
                # Inspect exception for quota/resource exhaustion and continue to next model
                msg = str(e).lower()
                if "resource_exhausted" in msg or "quota" in msg or "429" in msg:
                    # try next model candidate
                    continue
                # for other errors, stop and raise
                raise
        # if all candidates failed, raise last exception or return fallback
        if last_exc:
            raise last_exc
        return ("[FALLBACK] " + prompt)[: max(200, max_tokens)]

    def _generate_for_model(self, model_short: str, prompt: str, max_tokens: int = 256) -> str:
        """Attempt to generate using a specific model short name (e.g., 'gemini-2.5-flash')."""
        # Try SDK generation if available
        if self._genai is not None and self.api_key:
            if self._genai_name == "genai":
                try:
                    self._genai_client = self._genai.Client(api_key=self.api_key)
                except Exception:
                    self._genai_client = None
                if self._genai_client is not None:
                    cfg = {"max_output_tokens": max_tokens}
                    resp = self._genai_client.models.generate_content(model=model_short, contents=prompt, config=cfg)
                    data = getattr(resp, "model_dump", None) and resp.model_dump() or getattr(resp, "dict", lambda: {} )()
                    if isinstance(data, dict):
                        if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
                            cand = data["candidates"][0]
                            content = cand.get("content") or {}
                            parts = content.get("parts") if isinstance(content, dict) else None
                            if parts and isinstance(parts, list) and len(parts) > 0:
                                first = parts[0]
                                txt = first.get("text") or first.get("content")
                                if txt:
                                    return txt
                        if "text" in data and data.get("text"):
                            return data.get("text")
        # older google.generativeai SDK
        if self._genai is not None and self._genai_name == "generativeai":
            try:
                self._genai.configure(api_key=self.api_key)
            except Exception:
                pass
            if hasattr(self._genai, "generate"):
                resp = self._genai.generate(model=model_short, prompt=prompt, max_output_tokens=max_tokens)
                if isinstance(resp, dict):
                    return resp.get("text") or json.dumps(resp)
                text = getattr(resp, "text", None) or getattr(resp, "output", None)
                if text:
                    return text

        # HTTP REST call to model-specific generate endpoint
        if self.api_key and self.api_url:
            url = f"{self.api_url.rstrip('/')}/v1/models/{model_short}:generate"
            payload = {"prompt": {"text": prompt}, "maxOutputTokens": max_tokens}
            resp = self._post_with_key_fallback(url, payload, timeout=30)
            if resp is None:
                raise RuntimeError("No response from generate endpoint")
            if resp.status_code == 404:
                return ("[FALLBACK] " + prompt)[: max(200, max_tokens)]
            try:
                resp.raise_for_status()
            except Exception as e:
                # raise with response body for debugging
                body = None
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                raise RuntimeError(f"HTTP {resp.status_code}: {body}")
            data = resp.json()
            if isinstance(data, dict):
                if "candidates" in data and isinstance(data["candidates"], list):
                    c = data["candidates"][0]
                    # candidates may be dict with content or text
                    if isinstance(c, dict):
                        content = c.get("content") or c
                        if isinstance(content, dict) and "parts" in content:
                            parts = content.get("parts", [])
                            if parts:
                                return parts[0].get("text") or parts[0].get("content") or json.dumps(parts[0])
                        if "text" in c:
                            return c.get("text")
                    return json.dumps(c)
                if "outputs" in data and isinstance(data["outputs"], list):
                    out = data["outputs"][0]
                    if isinstance(out, dict):
                        return out.get("content") or json.dumps(out)
                if "text" in data:
                    return data.get("text")
            return json.dumps(data)

        # fallback local
        return ("[FALLBACK] " + prompt)[: max(200, max_tokens)]

        # HTTP REST call to model-specific generate endpoint
        if self.api_key and self.api_url:
            url = f"{self.api_url.rstrip('/')}/v1/models/{self._generate_model_short}:generate"
            # standard request body for Generative Language generate
            payload = {"prompt": {"text": prompt}, "maxOutputTokens": max_tokens}
            resp = self._post_with_key_fallback(url, payload, timeout=30)
            if resp is None:
                raise RuntimeError("No response from generate endpoint")
            if resp.status_code == 404:
                return ("[FALLBACK] " + prompt)[: max(200, max_tokens)]
            resp.raise_for_status()
            data = resp.json()
            # try to extract text from common response shapes
            if isinstance(data, dict):
                if "candidates" in data and isinstance(data["candidates"], list):
                    return data["candidates"][0].get("content") if data["candidates"] else json.dumps(data)
                if "outputs" in data and isinstance(data["outputs"], list):
                    # outputs may contain text blocks
                    out = data["outputs"][0]
                    if isinstance(out, dict):
                        return out.get("content") or json.dumps(out)
                if "text" in data:
                    return data.get("text")
            return json.dumps(data)

        # Simple local fallback: echoing with truncation
        return ("[FALLBACK] " + prompt)[: max(200, max_tokens)]

    def _deterministic_embedding(self, text: str, dim: int = 1536) -> List[float]:
        # Create a deterministic pseudo-embedding from the text hash
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(h[:8], "big") % (2 ** 31))
        vec = rng.normal(size=(dim,)).astype("float32")
        # normalize
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec.tolist()


class QAEngine:
    def __init__(self, gemini_client: GeminiClient, store):
        self.client = gemini_client
        self.store = store

    def _get_context(self, query: str, top_k: int = 5) -> str:
        q_emb = self.client.embed_texts([query])[0]
        hits = self.store.search(q_emb, top_k=top_k)
        context = "\n\n".join([h[0] for h in hits])
        return context

    def answer_question(self, question: str) -> str:
        context = self._get_context(question, top_k=5)
        prompt = f"Use the following context to answer the question. Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return self.client.generate_text(prompt)

    def summarize(self, topic: Optional[str] = None) -> str:
        # Summarize top documents or a topic-guided summary
        if topic:
            prompt = f"Summarize the document in the context of: {topic}. Provide concise bullets."
        else:
            prompt = f"Summarize the provided document into concise bullets."
        # use concatenated docs as context
        docs = self.store._metadatas[:10]
        prompt = f"Context:\n{'\n\n'.join(docs)}\n\n{prompt}"
        return self.client.generate_text(prompt)

    def generate_mcqs(self, topic: Optional[str] = None, n: int = 5) -> List[dict]:
        # Build a grounded, structured prompt asking for JSON output to reduce hallucinations.
        context = "\n\n".join(self.store._metadatas[:10])
        prompt = (
            "You are given the following CONTEXT extracted from a document. Create up to {n} multiple-choice "
            "questions (4 options each) strictly based on the CONTEXT. Do NOT invent facts or add information "
            "that is not present in the CONTEXT. If the answer cannot be determined from the CONTEXT, use the "
            "string \"UNKNOWN\" for the answer.\n\n"
        ).format(n=n)
        prompt += "CONTEXT:\n" + context + "\n\n"
        prompt += (
            "Return the result as a JSON array of objects with keys: question (string), options (array of 4 strings), "
            "answer (the exact option text that is correct). Example: [{\"question\":\"...\",\"options\":[\"A\",\"B\",\"C\",\"D\"],\"answer\":\"A\"}].\n"
        )
        prompt += "Only output valid JSON. Do not add any commentary.\n"

        out = self.client.generate_text(prompt, max_tokens=800)
        # Try to parse JSON strictly. If parsing fails, attempt to extract a JSON substring.
        try:
            parsed = json.loads(out)
            # Normalize to expected shape and cap to n
            if isinstance(parsed, list):
                return parsed[:n]
        except Exception:
            # attempt to find JSON inside text
            try:
                import re

                m = re.search(r"(\[\s*\{.*\}\s*\])", out, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(1))
                    if isinstance(parsed, list):
                        return parsed[:n]
            except Exception:
                pass

        # Fallback: return a single question constructed from the top of the output but mark answer UNKNOWN
        snippet = out.strip().splitlines()
        if snippet:
            q = snippet[0][:200]
        else:
            q = "Generate MCQs"
        return [{"question": q, "options": ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"], "answer": "UNKNOWN"}]

    def generate_notes(self) -> str:
        prompt = "Create concise study notes from the following document. Use headings and bullet points." + "\n\n"
        prompt += "\n\n".join(self.store._metadatas[:20])
        return self.client.generate_text(prompt, max_tokens=800)
