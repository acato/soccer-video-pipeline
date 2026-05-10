"""Smoke test the v6 c757 vLLM endpoint with a classifier-flavored prompt.

Verifies the model produces non-empty content (not just HTTP 200).
"""
import base64, io, json, urllib.request, urllib.error
from PIL import Image

img = Image.new("RGB", (336, 336), (50, 120, 200))
buf = io.BytesIO(); img.save(buf, "JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

prompt = (
    "You are analyzing 1 frame from a soccer match (0s - 5s). "
    "Reply with a JSON array. Each element: "
    '{"event_type":"...","start_sec":N,"end_sec":N,"confidence":0.0-1.0}. '
    "If no events seen, reply with []."
)
payload = {
    "model": "qwen3-vl-32b",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}],
    "max_tokens": 200,
    "temperature": 0,
}

req = urllib.request.Request(
    "http://10.10.2.222:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
    choice = resp["choices"][0]
    msg = choice["message"]
    usage = resp["usage"]
    ct = usage["completion_tokens"]
    fr = choice["finish_reason"]
    sr = choice.get("stop_reason")
    c = msg["content"] or ""
    print(f"completion_tokens={ct} finish_reason={fr} stop_reason={sr}")
    print(f"CONTENT[{len(c)} chars]:")
    print(c[:400])
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read().decode()[:600])
except Exception as e:
    print("ERR", type(e).__name__, e)
