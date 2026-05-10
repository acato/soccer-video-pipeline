"""Smoke test the vLLM endpoint with a small synthetic JPEG, mimicking
the Mac pipeline's request shape (5-frame sliding window).

Run anywhere with internet access to the LLM server's HTTP endpoint."""
import base64, io, json, sys
from urllib import request

# Tiny 1280x550 RGB image (similar to what Mac extracts from video frames)
try:
    from PIL import Image
except ImportError:
    print("install PIL first: pip install Pillow")
    sys.exit(1)

img = Image.new("RGB", (1280, 550), (50, 90, 30))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=70)
b64 = base64.b64encode(buf.getvalue()).decode()
data_uri = f"data:image/jpeg;base64,{b64}"

# Build a chat completion with 5 frames (matches Mac pipeline window size)
payload = {
    "model": "qwen3-vl-32b",
    "messages": [{
        "role": "user",
        "content": [
            *[{"type": "image_url", "image_url": {"url": data_uri}} for _ in range(5)],
            {"type": "text", "text": "Describe what you see in one word."}
        ]
    }],
    "max_tokens": 20,
    "temperature": 0,
}

req = request.Request(
    "http://10.10.2.222:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with request.urlopen(req, timeout=60) as r:
        result = json.loads(r.read().decode())
        print("STATUS: OK")
        print("response:", result["choices"][0]["message"]["content"])
        print("usage:", result["usage"])
except Exception as e:
    print("STATUS: FAIL")
    print("error:", e)
    if hasattr(e, "read"):
        print("body:", e.read().decode()[:500])
