import os, re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 6)]
for i, k in enumerate(keys, 1):
    if not k:
        print(f"Key {i}: NOT SET")
        continue
    try:
        Groq(api_key=k).chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        print(f"Key {i}: OK")
    except Exception as e:
        err = str(e)
        org = re.search(r"org_\w+", err)
        tpd = re.search(r"Used (\d+)", err)
        print(f"Key {i}: FAIL  org={org.group() if org else '?'}  used={tpd.group(1) if tpd else '?'}  | {err[:100]}")
