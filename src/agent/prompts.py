"""
src/agent/prompts.py
Prompt templates cho ReAct Agent.
"""

# # ── Prompt phân tích câu hỏi → chọn tool ─────────────────────

# ROUTER_PROMPT = """You are an IT helpdesk assistant. Analyze the user question and decide which tool to use.

# Tools:
# 1. CYPHER - Use ONLY when question contains specific error codes (like 0x..., ERROR_XXX, KB numbers) or exact product names
# 2. EMBEDDING - Use when question is descriptive or vague (like "not working", "fails", "cannot connect")
# 3. BFS - Use when question asks about relationship or cause between TWO specific things (like "what causes X when Y")
# 4. WEBSEARCH - Use when question mentions recent updates, latest issues, or very specific version numbers

# Examples:
# - "How to fix ERROR_INVALID_HANDLE?" → CYPHER
# - "My smart card is not working" → EMBEDDING
# - "What causes Teams to fail when VPN is on?" → BFS
# - "Windows 11 24H2 update issue" → WEBSEARCH
# - "Cannot connect to internet after update" → EMBEDDING
# - "Teams meeting keeps dropping" → EMBEDDING

# Question: {question}

# Reply with ONLY one word: CYPHER, EMBEDDING, BFS, or WEBSEARCH"""


# # ── Prompt trích xuất entity từ câu hỏi ──────────────────────

# ENTITY_EXTRACT_PROMPT = """Extract the main IT entity or error from this question.
# Return ONLY the entity name, nothing else.

# Examples:
# Question: "How to fix ERROR_INVALID_HANDLE?" → ERROR_INVALID_HANDLE
# Question: "Smart card not working" → Smart Card
# Question: "Teams meeting fails" → Teams Meeting Failure
# Question: "Cannot connect to internet" → Network Connection
# Question: "Windows update failing" → Windows Update

# Question: {question}
# Entity:"""


# # ── Prompt sinh câu trả lời cuối ─────────────────────────────

# ANSWER_PROMPT = """You are an IT helpdesk assistant. Answer the user question based on the context below.
# Be concise, practical, and include step-by-step instructions if available.
# If the context doesn't have enough information, say so honestly and provide general troubleshooting steps.

# User Question: {question}

# Context from Knowledge Base:
# {context}

# Answer:"""


# # ── Prompt cho BFS (2 entities) ──────────────────────────────

# BFS_ENTITY_PROMPT = """Extract TWO IT entities from this question for relationship analysis.
# Return ONLY two entity names separated by | symbol. No labels, no explanation.

# Example:
# Input: "What causes Teams to fail when VPN is on?"
# Output: Teams Meeting Failure | VPN Connection

# Input: "How does Windows Update affect network?"
# Output: Windows Update | Network Connection

# Input: {question}
# Output:"""


# ── Prompt phát hiện câu hỏi mơ hồ ──────────────────────────

IS_AMBIGUOUS_PROMPT = """Does this IT question REQUIRE prior conversation context to be understood?
Reply "yes" ONLY if the question cannot be answered without knowing what was discussed before — for example, a bare "how do I fix it?" where "it" has no referent inside the question itself.
Reply "no" if the question names a specific device, error, symptom, or topic — even if it also uses words like "it" or "this".

Examples:
- "how do I fix it?" → yes  (no subject stated)
- "my wifi is slow, how to fix it?" → no  (wifi is the subject)
- "my computer is slow, how to fix it?" → no  (computer is the subject)
- "what about VPN?" → yes  (refers to prior topic)
- "Teams meeting keeps dropping" → no  (self-contained)
- "same issue on my laptop" → yes  (refers to prior issue)

Reply ONLY yes or no.

Question: {question}"""


# ── Prompt phát hiện topic thay đổi ──────────────────────────

TOPIC_CHANGE_PROMPT = """Compare these two questions. Are they about completely different IT topics with no relation?

Examples of topic change (yes):
- Previous: "Smart card not working" / Current: "Windows update failing" → yes
- Previous: "Teams meeting drops" / Current: "Cannot print document" → yes
- Previous: "VPN connection issue" / Current: "Email not sending" → yes

Examples of same topic (no):
- Previous: "Teams meeting drops" / Current: "What about VPN?" → no
- Previous: "Smart card error" / Current: "How do I fix it?" → no
- Previous: "VPN issue" / Current: "How do I can fix this issue?" → no
- Previous: "Windows update error" / Current: "What should I do next?" → no

Previous: {prev}
Current: {current}

Reply ONLY yes or no."""


# ── Prompt planning (lightweight thinking note) ───────────────

PLAN_PROMPT = """You are an IT helpdesk assistant. Briefly analyze what the user needs.
In 1-2 sentences, describe: what type of problem this is and what information would best answer it.
Be concise — this is an internal thinking note, not shown to the user.

Question: {question}
Plan:"""


# ── Prompt reflection (self-evaluation) ──────────────────────

REFLECT_PROMPT = """Evaluate this IT helpdesk answer.

Question: {question}
Answer: {answer}
Sources used: {num_sources} source(s), tools: {tools_used}

Reply with ONLY valid JSON, no markdown:
{{"is_sufficient": true, "confidence": "high", "reason": "one sentence"}}

Confidence criteria:
- "high": clear, actionable answer with specific steps and sources
- "medium": partial info, generic steps, or only 1 weak source
- "low": no sources, vague answer, or answer does not address the question

is_sufficient: false only when the answer is completely off-topic or empty."""


# ── Merged preprocessing (ambiguity + topic-change) ───────────

PREPROCESS_PROMPT = """Analyze this IT helpdesk conversation turn.

Previous question: {prev}
Current question: {current}

Answer TWO questions:
1. is_ambiguous: Does the current question require prior context to be understood?
   - true ONLY if it has no discernible subject (e.g. bare "how do I fix it?", "what about it?")
   - false if it names any device, error, product, or symptom

2. topic_changed: Is the current question about a completely different IT topic?
   - true ONLY if topics are fully unrelated (e.g. "smart card" → "printing issue")
   - false for follow-up questions, elaborations, or loosely related topics

Reply ONLY with valid JSON, no markdown:
{{"is_ambiguous": false, "topic_changed": false}}

Rule: If {prev} is empty, always return: {{"is_ambiguous": false, "topic_changed": false}}"""