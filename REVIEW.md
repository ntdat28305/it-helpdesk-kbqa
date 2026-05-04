# Project Audit — IT Helpdesk KBQA

> Rà soát chi tiết toàn bộ codebase tại commit `fc6d9e5` (main branch).
> Mức độ ưu tiên: 🔴 Phải sửa | 🟡 Nên sửa | 🟢 Nice-to-have.

---

## Tóm tắt điều hành

| Khía cạnh | Trạng thái | Ghi chú |
|---|---|---|
| **Kiến trúc tổng thể** | ⚠️ Có nhưng chưa đúng claim | "Agentic AI" thực ra là LLM router + RAG, không có ReAct loop |
| **Knowledge Graph** | 🟡 Hoạt động nhưng chậm | Thiếu index hoàn toàn; bug logic trong query |
| **Agent core** | 🟡 Linear pipeline, có vài bug nhỏ | Không có self-reflection, không multi-tool |
| **FastAPI** | 🟡 Chạy được nhưng leak memory | Sessions tích lũy không bao giờ được dọn |
| **GCN training** | 🟡 OK về cấu trúc, đáng ngờ về data leak | Validate trên `train_data.x` thay vì `val_data.x` |
| **Evaluation** | 🔴 Không tin cậy được | Sample bias, test set circular, mapping sai |
| **Docker** | 🔴 Compose file đang lỗi | Trỏ vào `Dockerfile` không tồn tại |
| **Test/Lint** | 🔴 Không có | Thư mục `tests/` rỗng, không có CI |
| **Bảo mật** | 🟡 Có 1 chỗ injection nhẹ | KG loader f-string relation type |

---

## 1. Kiến trúc & "Agentic AI" claim

### 🔴 1.1. Không phải agentic AI thực sự

[src/agent/agent.py:254-400](src/agent/agent.py#L254-L400) gọi là "ReAct Agent" trong docstring, nhưng `answer()` là **pipeline tuyến tính 1-pass**:

```
question → router (1 LLM) → entity extract (1 LLM) → 1 tool call
        → [fallback WEBSEARCH 1 lần nếu rỗng] → answer (1 LLM)
```

Không có vòng lặp Thought–Action–Observation, không có multi-tool composition, không có self-reflection. Câu hỏi cần kết hợp 2 tool (vd: "Lỗi A và lỗi B có liên quan không và làm sao fix cả 2?") là không xử lý được.

**Hệ quả:** nếu báo cáo/slide ghi "Agentic AI" hoặc "ReAct", hội đồng có thể chất vấn. Cách gọi chính xác hơn:
- "LLM-routed Retrieval-Augmented Generation"
- "Tool-augmented QA with intent classification"

**Fix triệt để:** viết lại `answer()` thành vòng `for step in range(max_steps): ...` với điều kiện dừng (LLM tự quyết "FINISH"), dùng Groq function-calling API (xem [§2.4](#-24-không-dùng-function-callingstructured-output)).

### 🟡 1.2. Schema KG quá đơn giản cho bài toán

Hiện chỉ có:
- `(:Entity {name, type})` — `type` không được dùng ở agent
- `(:Article {article_id, title, url, category})`
- `(a:Article)-[:MENTIONS]->(e:Entity)`
- `(e1:Entity)-[REL_TYPE]->(e2:Entity)` với `REL_TYPE` ∈ {CAUSES, FIXES, AFFECTS, REQUIRES, RELATED_TO}

Vấn đề:
- Edge không có properties (confidence, source article, extracted_at) — không thể trace lại nguồn của relation
- `Entity.type` được lưu ở loader nhưng **agent không bao giờ filter theo type**, lãng phí
- Không có `Symptom`/`Solution` node riêng → mọi thứ là `Entity` → CYPHER không phân biệt được "đây là lỗi" vs "đây là cách fix"

### 🟢 1.3. Tách riêng `community.json` và `community_summaries.json`

[src/kg_build/community.py:107-115](src/kg_build/community.py#L107-L115) lưu `{comm_id: [node_names]}`. Nhưng agent ở [src/agent/agent.py:367](src/agent/agent.py#L367) đọc `info.get("nodes", [])` → expect `{comm_id: {"nodes": [...], "summary": "..."}}`. Tức là `summarizer.py` (không đọc trong audit này) phải transform giữa 2 format. Đây là coupling ngầm dễ vỡ — nên unify schema ngay từ `community.py`.

---

## 2. Module Agent — `src/agent/agent.py`

### 🔴 2.1. Bug: `display_entity` dùng `dir()` để check biến local

[src/agent/agent.py:388](src/agent/agent.py#L388):
```python
display_entity = e1 if tool == "BFS" and 'e1' in dir() else entity
```

`dir()` không có argument trả về **tên trong namespace hiện tại** — đúng là chứa `e1` nếu đã gán. Nhưng:
1. Cách viết phi-Pythonic, dễ false positive (nếu `e1` được gán ở scope ngoài thì cũng pass)
2. Khi `tool` đã bị overwrite thành `"WEBSEARCH"` qua fallback ([agent.py:361](src/agent/agent.py#L361)) thì `e1` vẫn tồn tại nhưng không liên quan → vẫn dùng `e1` làm display, sai
3. Nếu BFS rẽ vào nhánh `else` ở [agent.py:348-350](src/agent/agent.py#L348-L350) (chỉ có e1, không có e2), `tool` đổi thành `"CYPHER"` → check `tool == "BFS"` fail → dùng `entity` đúng

**Fix:**
```python
display_entity = entity  # mặc định dùng kết quả ENTITY_EXTRACT_PROMPT
```
hoặc nếu thật sự cần entity từ BFS, gán `display_entity` ngay trong nhánh BFS.

### 🔴 2.2. Bug: silent fallback khi LLM trả tool không hợp lệ

[src/agent/agent.py:283-285](src/agent/agent.py#L283-L285):
```python
valid_tools = {"CYPHER", "EMBEDDING", "BFS", "WEBSEARCH"}
if tool not in valid_tools:
    tool = "EMBEDDING"
```

LLM trả "I think we should use CYPHER" → `tool = "I THINK..."` → fallback EMBEDDING. **Không log warning**, không retry. Nguyên nhân chính khiến debug routing khó.

**Fix:**
```python
if tool not in valid_tools:
    logger.warning(f"Invalid tool '{tool}' from LLM, fallback to EMBEDDING")
    tool = "EMBEDDING"
```
Hoặc tốt hơn: dùng Groq function-calling API để LLM bắt buộc trả structured output.

### 🟡 2.3. Bug nhỏ: `from unittest import result` rác

[src/agent/agent.py:20](src/agent/agent.py#L20):
```python
from unittest import result
```
Import này **không bao giờ được dùng**. Có vẻ là auto-import của IDE khi gõ biến `result`. Xóa.

### 🟡 2.4. Không dùng function-calling/structured output

Toàn bộ tool routing và entity extraction đều parse text từ LLM:
- `ROUTER_PROMPT` → `.upper().strip()` để lấy 1 từ
- `BFS_ENTITY_PROMPT` → `.split("|")` để lấy 2 entity
- `IS_AMBIGUOUS_PROMPT` → check `.startswith("yes")`
- `TOPIC_CHANGE_PROMPT` → check `.startswith("yes")`

**Vấn đề:** LLM ở `temperature=0` cũng vẫn có thể thêm prefix/suffix ("Tool: CYPHER", "**EMBEDDING**", "Yes, this is ambiguous"). Mỗi parsing pattern là 1 điểm fragile.

**Fix:** Groq SDK hỗ trợ `tools=[...]` với JSON schema chuẩn. Định nghĩa schema 1 lần, LLM bắt buộc trả JSON đúng format.

### 🟡 2.5. Conversation history quá đơn giản

[src/agent/agent.py:386-387](src/agent/agent.py#L386-L387):
```python
if len(self.history) > 20:
    self.history = self.history[-20:]
```

Cứng nhắc cắt 20 message — **mất hoàn toàn context cũ** thay vì summarize. Với câu hỏi dài liên quan đến nhiều turn, history bị clip giữa câu.

**Fix:** Khi vượt threshold, gọi LLM summarize 10 message cũ thành 1 system message rồi giữ. Pattern phổ biến là "rolling summary".

### 🟡 2.6. `llm_call` âm thầm trả `""` khi lỗi

[src/agent/agent.py:81-93](src/agent/agent.py#L81-L93):
```python
def llm_call(...):
    try:
        ...
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return ""
```

Trả `""` khi lỗi → mọi caller phía dưới sẽ:
- Router: `"".upper() not in valid_tools` → fallback EMBEDDING (im lặng)
- Entity extract: `entity = ""` → CYPHER với empty string → match toàn bộ KG ([§3.2](#-32-bug-empty-entity_name--match-mọi-entity))
- Topic change: `"".startswith("yes")` = False → giữ history
- Ambiguous: tương tự

**Fix:** raise exception hoặc return `None`, để caller tự quyết định fallback.

### 🟢 2.7. `embedding_search` threshold cứng = 50

[src/agent/agent.py:111-114](src/agent/agent.py#L111-L114):
```python
match_result = fuzz_process.extractOne(query_entity, node_names)
if not match_result or match_result[1] < 50:
    return []
```

Threshold 50 (rapidfuzz dùng thang 0–100) là khá thấp, có thể cho phép "VPN" match "VPNet" nhầm, hoặc ngược lại từ chối "Smartcard" vs "Smart Card" (có thể chỉ ~75). Nên test empirically và tách thành config.

### 🟢 2.8. Resources load 1 lần ở `__init__`, không hot-reload

Khi train lại GCN, agent **không tự load embeddings mới** — phải restart toàn bộ FastAPI. Trong context demo này có thể chấp nhận, nhưng ghi vào docs.

### 🟡 2.9. Comment tiếng Việt indent sai

[src/agent/agent.py:257-262](src/agent/agent.py#L257-L262):
```python
        # Fix 5: Tự động phát hiện topic change → reset history
        # Fix 5: Tự động phát hiện topic change → reset history
# Chỉ check khi câu hỏi KHÔNG mơ hồ
        if self.history:
```
Comment thứ 3 (`# Chỉ check...`) không có indent → trông như module-level comment xen vào method body. Cosmetic nhưng làm method khó đọc.

---

## 3. CYPHER / Neo4j Query — `src/agent/neo4j_query.py`

### 🔴 3.1. Bug: `WHERE` chỉ check entity ở slot `e`, bỏ qua `e2`

[src/agent/neo4j_query.py:35-44](src/agent/neo4j_query.py#L35-L44):
```cypher
MATCH (e:Entity)-[r]-(e2:Entity)
WHERE toLower(e.name) CONTAINS toLower($name)
RETURN e.name AS src, type(r) AS rel, e2.name AS tgt
LIMIT 20
```

Pattern undirected `(e)-[r]-(e2)` Neo4j sẽ **bind cả 2 chiều** — nghĩa là với 1 cạnh `(VPN)-[:CAUSES]->(Teams)` sẽ có 2 row: (e=VPN, e2=Teams) và (e=Teams, e2=VPN). Nhưng `WHERE` chỉ filter trên `e`, nên khi user search "Teams", row có e=VPN sẽ bị loại — mặc dù row đó chứa "Teams" ở `e2`.

Thực tế Cypher engine sẽ vẫn match row (e=Teams, e2=VPN) → kết quả không sai hoàn toàn. **Nhưng** với `LIMIT 20`, các row mà entity nằm ở slot bind ngược sẽ chiếm chỗ của các row hợp lệ.

**Fix:**
```cypher
MATCH (e:Entity)-[r]-(e2:Entity)
WHERE toLower(e.name) CONTAINS toLower($name)
   OR toLower(e2.name) CONTAINS toLower($name)
WITH CASE WHEN toLower(e.name) CONTAINS toLower($name) THEN e ELSE e2 END AS center,
     CASE WHEN toLower(e.name) CONTAINS toLower($name) THEN e2 ELSE e END AS other,
     r
RETURN center.name AS src, type(r) AS rel, other.name AS tgt
LIMIT 20
```

### 🔴 3.2. Bug: empty `entity_name` → match mọi entity

`CONTAINS ""` luôn `true` trong Cypher → nếu LLM extract fail và `entity_name = ""`, query trả 20 entity ngẫu nhiên + 5 article ngẫu nhiên làm context. Agent không biết, vẫn sinh answer dựa trên context rác.

**Fix:** đầu cả `cypher_search` và `bfs_search`:
```python
if not entity_name or not entity_name.strip():
    return {"entity": entity_name, "relations": [], "articles": []}
```

### 🔴 3.3. Thiếu hoàn toàn INDEX/CONSTRAINT

[src/kg_build/kg_loader.py](src/kg_build/kg_loader.py) **không tạo bất kỳ index hay constraint nào**. Hậu quả:

1. Mọi `MATCH (e:Entity) WHERE ... CONTAINS ...` → full-scan 3266 nodes
2. `MERGE (e:Entity {name: $name})` trong loader → cũng full-scan để check duplicate → loader chậm O(n²)
3. Mỗi câu user có thể trigger 4–5 query CYPHER (tool trực tiếp + nested khi EMBEDDING fetch sources cho top-3 nodes + BFS endpoint resolution + community lookup) → cộng dồn latency

**Fix tối thiểu** — chèn vào `kg_loader.py` ngay sau `verify_connectivity`:
```python
with driver.session() as session:
    session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE")
    session.run("CREATE CONSTRAINT article_id IF NOT EXISTS "
                "FOR (a:Article) REQUIRE a.article_id IS UNIQUE")
    session.run("CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name]")
```

### 🔴 3.4. Driver không phải singleton — mỗi query tạo connection mới

[src/agent/neo4j_query.py:16-20](src/agent/neo4j_query.py#L16-L20):
```python
def get_driver():
    return GraphDatabase.driver(...)
```
Và mỗi `cypher_search`/`bfs_search`/`get_graph_data` đều gọi `get_driver()` rồi `driver.close()`. Mỗi `GraphDatabase.driver(...)` mở **connection pool mới + TLS handshake** với AuraDB — đáng kể.

**Fix:** module-level singleton + đóng ở FastAPI lifespan shutdown:
```python
_driver = None
def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD")),
        )
    return _driver

def close_driver():
    global _driver
    if _driver: _driver.close(); _driver = None
```

### 🟡 3.5. `LIMIT` không có `ORDER BY` → kết quả non-deterministic

[src/agent/neo4j_query.py:35-56](src/agent/neo4j_query.py#L35-L56) cả 2 query đều `LIMIT N` không kèm `ORDER BY`. Neo4j trả row theo thứ tự bất kỳ (tùy storage engine + cache state). Cùng câu hỏi 2 lần có thể ra context khác nhau → answer khác nhau → khó debug, khó eval reproducible.

**Fix:** thêm `ORDER BY size(e.name) ASC, e.name` (ưu tiên match ngắn = chính xác hơn) hoặc rank theo số mention:
```cypher
MATCH (a:Article)-[:MENTIONS]->(e:Entity)
WHERE toLower(e.name) CONTAINS toLower($name)
WITH a, count(e) AS rel_count
RETURN a.title, a.url
ORDER BY rel_count DESC
LIMIT 5
```

### 🟡 3.6. Không có fuzzy match ở CYPHER (trong khi EMBEDDING có)

Inconsistency: agent dùng `rapidfuzz` trong `embedding_search` ([agent.py:111](src/agent/agent.py#L111)) nhưng CYPHER chỉ `CONTAINS` substring. LLM extract "Smartcard" mà KG có "Smart Card" → CYPHER miss. Có thể:
- Pre-normalize: `name.lower().replace(" ", "").replace("_", "")` ở cả lúc query và lúc lưu
- Hoặc dùng fulltext index Lucene với fuzzy: `"smartcard~"`

### 🟡 3.7. `bfs_search`: `shortestPath() ... LIMIT 3` vô nghĩa

[src/agent/neo4j_query.py:100-106](src/agent/neo4j_query.py#L100-L106):
```cypher
MATCH p = shortestPath((a)-[*..6]-(b))
RETURN [n in nodes(p) | n.name] AS path_nodes, length(p) AS path_length
LIMIT 3
```
Theo định nghĩa Cypher, `shortestPath()` chỉ trả **một** path duy nhất. `LIMIT 3` không có tác dụng. Để có nhiều path:
```cypher
MATCH p = allShortestPaths((a)-[*..6]-(b))
...
LIMIT 5
```

### 🟡 3.8. `bfs_search`: heuristic `size(name) ASC` chọn sai node

[src/agent/neo4j_query.py:78-82](src/agent/neo4j_query.py#L78-L82):
```cypher
WHERE toLower(e.name) CONTAINS toLower($name)
   OR toLower($name) CONTAINS toLower(e.name)
RETURN e.name AS name
ORDER BY size(e.name) ASC
LIMIT 1
```

User hỏi "Microsoft Teams" → nếu KG có cả node `Teams` (5 ký tự) và `Microsoft Teams` (15 ký tự), heuristic `ASC` sẽ chọn `Teams`. Sai semantic — node ngắn hơn ≠ match tốt hơn.

**Fix:** ưu tiên exact match → prefix match → contains:
```cypher
WHERE toLower(e.name) = toLower($name)
   OR toLower(e.name) STARTS WITH toLower($name)
   OR toLower(e.name) CONTAINS toLower($name)
ORDER BY
  CASE WHEN toLower(e.name) = toLower($name) THEN 0
       WHEN toLower(e.name) STARTS WITH toLower($name) THEN 1
       ELSE 2 END,
  size(e.name) ASC
LIMIT 1
```

### 🟡 3.9. `bfs_search` không kiểm `name1 == name2`

Nếu fuzzy match khiến cả `entity1` và `entity2` cùng resolve về 1 node, `shortestPath((a)-[*..6]-(b))` với a=b sẽ trả path length 0 (chính node đó). Vô nghĩa làm context.

### 🟢 3.10. 2 query thay vì 1

`cypher_search` chạy 2 round-trip (relations + articles) thay vì 1 query với `OPTIONAL MATCH`. Latency × 2. Có thể combine:
```cypher
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($name)
OPTIONAL MATCH (e)-[r]-(e2:Entity)
OPTIONAL MATCH (a:Article)-[:MENTIONS]->(e)
RETURN
  collect(DISTINCT {src: e.name, rel: type(r), tgt: e2.name})[..20] AS relations,
  collect(DISTINCT {title: a.title, url: a.url})[..5] AS articles
```

### 🟢 3.11. `get_community_context` O(n*m) string match

[src/agent/neo4j_query.py:121-136](src/agent/neo4j_query.py#L121-L136) lặp tất cả community × tất cả node mỗi câu hỏi. Với 3266 nodes và (giả sử) 50 communities → 163k string ops mỗi query. Không nghiêm trọng nhưng có thể cache map `entity_lower → best_community_id` ở `__init__`.

---

## 4. KG Loader — `src/kg_build/kg_loader.py`

### 🔴 4.1. Cypher injection nhẹ ở relation type

[src/kg_build/kg_loader.py:100-104](src/kg_build/kg_loader.py#L100-L104):
```python
relation = rel.get("relation", "RELATED_TO").strip().upper()
...
session.run(f"""
    MATCH (a:Entity {{name: $source}})
    MATCH (b:Entity {{name: $target}})
    MERGE (a)-[:{relation}]->(b)
""", source=source, target=target)
```

`relation` đến từ LLM, được `f-string` thẳng vào Cypher. `.upper()` không loại bỏ ký tự đặc biệt. Nếu LLM trả `"FOO]->(x) DETACH DELETE x //"`, có thể inject (mặc dù xác suất thấp với prompt hiện tại).

**Fix nhanh:**
```python
import re
if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", relation):
    relation = "RELATED_TO"
```

**Fix triệt để:** dùng APOC:
```cypher
CALL apoc.merge.relationship(a, $relation, {}, {}, b)
```

### 🔴 4.2. Không tạo INDEX/CONSTRAINT

Đã nêu ở [§3.3](#-33-thiếu-hoàn-toàn-indexconstraint). Nhắc lại vì điểm fix nằm ở loader.

### 🟡 4.3. Mỗi entity = 2 round-trip riêng

[src/kg_build/kg_loader.py:75-85](src/kg_build/kg_loader.py#L75-L85):
```python
session.run("MERGE (e:Entity {name: $name}) SET e.type = $type", ...)
session.run("MATCH ... MATCH ... MERGE (a)-[:MENTIONS]->(e)", ...)
```
2 query/entity, không batch. Có thể UNWIND danh sách entities thành 1 query:
```cypher
UNWIND $entities AS ent
MERGE (e:Entity {name: ent.name})
SET e.type = ent.type
WITH e, ent
MATCH (a:Article {article_id: $article_id})
MERGE (a)-[:MENTIONS]->(e)
```

### 🟡 4.4. Loader không idempotent với entity rename

`MERGE (e:Entity {name: $name})` dùng `name` làm primary key. Nếu LLM extract tên sai chính tả ở 1 article rồi đúng ở article khác (vd "Windowws Update" vs "Windows Update"), KG có **2 node trùng lặp** không link. Cần normalize tên trước khi MERGE (lowercase/strip/whitespace).

### 🟡 4.5. Không validate relation source/target tồn tại

[src/kg_build/kg_loader.py:100-104](src/kg_build/kg_loader.py#L100-L104) dùng `MATCH (a:Entity)` rồi `MATCH (b:Entity)` rồi `MERGE`. Nếu source/target chưa được MERGE ở vòng entities (LLM trả relation reference đến entity không có trong list), 2 MATCH fail im lặng → relation bị bỏ qua **không có log**.

**Fix:**
```python
result = session.run("""
    MATCH (a:Entity {name: $source})
    MATCH (b:Entity {name: $target})
    MERGE (a)-[r:%s]->(b)
    RETURN r
""" % relation, source=source, target=target)
if not result.single():
    logger.warning(f"Relation {source} -[{relation}]-> {target}: entity not found")
```

### 🟡 4.6. Một file = một transaction implicit, không atomic

Loop trong session chạy `session.run(...)` từng query → mỗi query là implicit auto-commit transaction (Python driver default). Nếu fail giữa chừng, KG bị half-loaded cho article đó. Nên wrap mỗi article trong `session.execute_write(...)` để rollback được.

---

## 5. Entity Extractor — `src/kg_build/entity_extractor.py`

### 🟡 5.1. `text[:2000]` cắt giữa chừng

[src/kg_build/entity_extractor.py:124](src/kg_build/entity_extractor.py#L124):
```python
text = text[:2000]
```
Cắt đúng 2000 ký tự — có thể giữa câu, giữa từ. LLM nhận input lủng củng → extraction kém chất lượng. Nên cắt ở ranh giới câu (`.split(". ")` rồi accumulate đến 2000) hoặc dùng tokenizer thật.

### 🟡 5.2. Không retry khi parse JSON fail

[src/kg_build/entity_extractor.py:139-141](src/kg_build/entity_extractor.py#L139-L141):
```python
except json.JSONDecodeError:
    logger.warning("Không parse được JSON từ response")
    return {}
```
JSON malformed (rất thường gặp với llama-3.1-8b-instant) → return `{}` luôn → article bị skip không có entity. Với 378 article × tỉ lệ fail 10–15% có thể mất 50+ article.

**Fix:**
- Thử retry với prompt "Return ONLY valid JSON, không có ký tự khác"
- Hoặc dùng `json_repair` library
- Hoặc dùng Groq's `response_format={"type": "json_object"}` mode (nếu llama-3.1-8b-instant hỗ trợ)

### 🟡 5.3. Checkpoint file dạng plain text dễ corrupt

[src/kg_build/entity_extractor.py:85-95](src/kg_build/entity_extractor.py#L85-L95):
```python
with open(CHECKPOINT, "a", encoding="utf-8") as f:
    f.write(file_id + "\n")
```
Mỗi file_id 1 dòng, append. OK cho idempotent rerun, nhưng:
- Nếu interrupt giữa lúc write → dòng cuối có thể partial → set load lên thiếu newline → file_id sai
- Không track timestamp/status → không biết file nào fail vs success

**Fix:** dùng JSON với `{file_id: "success" | "failed", ts: ...}` hoặc SQLite.

### 🟡 5.4. `time.sleep(2)` hard-coded sau mỗi article

[src/kg_build/entity_extractor.py:218](src/kg_build/entity_extractor.py#L218): 378 article × 2s = 12.6 phút overhead mỗi lần chạy full pipeline. Nên dùng adaptive rate limiting (chỉ sleep khi gặp 429) hoặc bỏ vì `GroqRotator` đã handle rate limit.

### 🟢 5.5. Prompt không enforce entity type vocabulary

`PROMPT_TEMPLATE` ở [entity_extractor.py:107-119](src/kg_build/entity_extractor.py#L107-L119) liệt kê `Error|Product|Fix|Symptom|Concept` nhưng không enforce. LLM có thể trả type khác (vd "OperatingSystem", "Service") → KG có type không đồng nhất → khó query về sau.

---

## 6. GCN Training — `src/embedding/train_gcn.py`

### 🔴 6.1. Validation dùng `train_data.x` thay vì `val_data.x`

[src/embedding/train_gcn.py:202-203](src/embedding/train_gcn.py#L202-L203):
```python
with torch.no_grad():
    z_val, val_recon_pos = model(train_data.x, val_data.edge_index)
```

Validation forward pass đang **dùng feature tensor của train** kết hợp với edge index của val. Đây là quirk vì cả train_data và val_data dùng cùng features (cả 2 đều `Data(x=features, ...)` ở [train_gcn.py:107-108](src/embedding/train_gcn.py#L107-L108)) — nên technically không phải "data leak" nặng. **Nhưng:**

- `val_data.edge_index` chỉ chứa val edges, được dùng làm input của GCN propagate trong forward → tức là mô hình thấy val edges để encode → **leak edge structure**
- Đúng ra phải dùng `train_data.edge_index` cho propagate, sau đó dùng `val_data.edge_index` chỉ cho decode (link prediction):
```python
z_val = model.encode(train_data.x, train_data.edge_index)  # encode chỉ dùng train edges
val_recon_pos = model.decode(z_val, val_data.edge_index)   # predict trên val edges
```

Đây là bug **GAE link prediction kinh điển** — best val_loss bị over-estimate, embedding output có thể ổn nhưng metric val không tin được.

### 🟡 6.2. Negative sampling không loại trừ positive edges

[src/embedding/train_gcn.py:184-186](src/embedding/train_gcn.py#L184-L186):
```python
neg_src = torch.randint(0, num_nodes, (num_neg,))
neg_tgt = torch.randint(0, num_nodes, (num_neg,))
```
Random uniform — có thể trùng vào positive edge thật → negative label sai. PyG có `torch_geometric.utils.negative_sampling` xử lý đúng.

### 🟡 6.3. Negative sampling cũng có self-loop

`neg_src == neg_tgt` không được loại → 1/N samples là self-edge label 0, nhưng GCN encode self-loop mặc định (`add_self_loops=True` trong GCNConv) → confusing signal.

### 🟡 6.4. `import features` thừa

[src/embedding/train_gcn.py:12](src/embedding/train_gcn.py#L12):
```python
from pyexpat import features
```
Không dùng. Có vẻ IDE auto-import từ `pyexpat`. Xóa.

### 🟡 6.5. `from sentence_transformers import SentenceTransformer` 2 lần

[src/embedding/train_gcn.py:17](src/embedding/train_gcn.py#L17) và [src/embedding/train_gcn.py:27](src/embedding/train_gcn.py#L27). Xóa 1 dòng.

### 🟡 6.6. `sentence-transformers` không có trong `requirements.txt`

Đọc requirements.txt: không thấy `sentence-transformers`, `torch`, `torch-geometric`, `graspologic`, `tavily-python`, `rank-bm25`. Tức là user clone repo + `pip install -r requirements.txt` sẽ **không chạy được training/eval/agent**.

**Fix:** thêm vào requirements:
```
torch>=2.1.0
torch-geometric>=2.4.0
sentence-transformers>=2.7.0
graspologic>=3.4.0
tavily-python>=0.3.0
rank-bm25>=0.2.2
```

### 🟢 6.7. Dropout ở val phase

`model.eval()` đã tắt dropout ([train_gcn.py:201](src/embedding/train_gcn.py#L201)) — OK, nhưng đảm bảo lúc save embedding cũng `model.eval()` (đã làm ở [line 240](src/embedding/train_gcn.py#L240)). Tốt.

### 🟢 6.8. Không có test set thực sự

Chỉ có train/val (90/10), không có test. Eval metric (Hit@k, MRR) hiện đang đo trên external test set ([data/test_set.json](data/test_set.json)) — đây không phải test edges của GCN. Nên thêm test split để có "GCN link prediction AUC" làm sanity check riêng cho phần embedding.

### 🟡 6.9. `requirements.txt` bị encoding UTF-16

Đọc file thấy mỗi ký tự có khoảng trắng giữa — đây là dấu hiệu file lưu UTF-16 thay vì UTF-8. Trên Linux/Mac có thể bị `pip` parse fail. Convert sang UTF-8 (no BOM).

---

## 7. Community Detection — `src/kg_build/community.py`

### 🔴 7.1. Format output không khớp với format agent đọc

[src/kg_build/community.py:111](src/kg_build/community.py#L111):
```python
data = {str(k): v for k, v in communities.items()}
# data[comm_id] = [node_name, ...]
```

Nhưng [src/agent/agent.py:367](src/agent/agent.py#L367) đọc:
```python
nodes = [n.lower() for n in info.get("nodes", [])]
```
Tức là agent expect `data[comm_id] = {"nodes": [...], "summary": "..."}`.

**Có khả năng `summarizer.py` (chưa đọc trong audit này) làm transform** — nhưng nếu vậy, community.py output không bao giờ được dùng trực tiếp. Hai file output có schema khác nhau hoàn toàn nhưng cùng đặt ở `data/`. Confusing.

**Fix:** unify ngay từ `community.py`:
```python
data = {
    str(k): {"nodes": v, "summary": ""}
    for k, v in communities.items()
}
```
Rồi `summarizer.py` chỉ fill field `summary`.

### 🟡 7.2. Edge directed `(a)-[r]->(b)` cho Leiden

[src/kg_build/community.py:40-43](src/kg_build/community.py#L40-L43):
```cypher
MATCH (a:Entity)-[r]->(b:Entity)
RETURN elementId(a) AS src, elementId(b) AS tgt
```
Leiden algorithm thường chạy trên đồ thị undirected. Nếu KG có cạnh A→B (CAUSES) và B→A (FIXES), 2 cạnh đếm thành weight 2 (hoặc 1 nếu duplicate elimination). Nên dùng `(a)-[r]-(b)` undirected hoặc dedupe pair `(min(s,t), max(s,t))`.

### 🟡 7.3. `leiden(edges, trials=3)` — `trials=3` rất ít

Leiden là stochastic. `trials=3` cho graph 3000+ nodes là không đủ stable. Default thường 10–20. Nếu chạy 2 lần, `data/communities.json` có thể khác hẳn.

---

## 8. API — `src/api/main.py`

### 🔴 8.1. Memory leak: sessions tích lũy vô hạn

[src/api/main.py:81](src/api/main.py#L81):
```python
sessions: dict[str, ITHelpdeskAgent] = {}
```

Mỗi `session_id` mới → tạo 1 `ITHelpdeskAgent` mới ([api/main.py:87](src/api/main.py#L87)). Mỗi agent load:
- `node_embeddings.npy` (3266 × 32 × 4 bytes ≈ 418 KB)
- `community_summaries.json` (~vài KB)
- Groq client + history list

Trong môi trường demo nhỏ thì OK, **nhưng có 2 vấn đề**:
1. **Không có eviction** — chạy lâu, sessions = `{"abc", "def", ..., 10000}` → vài GB RAM
2. **`DELETE /session/{id}` chỉ reset history**, không xóa khỏi dict ([api/main.py:133-140](src/api/main.py#L133-L140))

**Fix:**
- Dùng `cachetools.TTLCache(maxsize=1000, ttl=3600)` cho sessions
- `DELETE /session/{id}` phải `del sessions[id]`
- Hoặc lift embeddings lên app-level, agent chỉ giữ history

### 🟡 8.2. Global `agent` được tạo nhưng không dùng

[src/api/main.py:25, 33](src/api/main.py#L25-L33) tạo global `agent` ở `lifespan` startup. Nhưng `/query` endpoint dùng `get_or_create_session(...)` không bao giờ touch `agent` này. Code dead.

**Fix:** xóa global `agent`, hoặc dùng nó làm "shared resources holder" (embeddings, communities) — các session agent chỉ giữ history, mượn resources từ global.

### 🟡 8.3. CORS `allow_origins=["*"]` quá lỏng

[src/api/main.py:48-52](src/api/main.py#L48-L52). Cho production cần whitelist UI domain. Demo thì OK nhưng nên ghi chú.

### 🟡 8.4. Không có rate limiting / auth

Bất cứ ai biết URL `/query` đều có thể spam → tốn Groq quota của project. Demo OK, nhưng deploy public phải thêm `slowapi` hoặc API key check.

### 🟡 8.5. `timeout=60` hard-coded ở UI gọi API

[src/ui/app.py:46](src/ui/app.py#L46) — agent có thể thực sự chạy >60s nếu Neo4j chậm + 4–6 LLM call. UI sẽ throw `ReadTimeout` mà user thấy "Connection error" → trải nghiệm tệ.

### 🟢 8.6. Lifespan không close Neo4j driver

`lifespan` shutdown chỉ log "Shutting down..." không đóng driver. Combined với issue [§3.4](#-34-driver-không-phải-singleton--mỗi-query-tạo-connection-mới) thì hiện tại không leak (driver đóng sau mỗi call), nhưng nếu fix singleton thì phải nhớ đóng ở shutdown.

---

## 9. UI — `src/ui/app.py`

### 🟡 9.1. Session ID 8 ký tự uuid — đụng độ

[src/ui/app.py:25](src/ui/app.py#L25):
```python
str(uuid.uuid4())[:8]
```
8 hex char = 32 bit ≈ 4 tỷ space. Birthday paradox → 50% đụng độ ở ~65k session. Nhỏ hơn nhiều sessions limit. Trong demo dùng tốt, nhưng **2 user trùng session → share history** (privacy issue).

**Fix:** dùng full uuid hoặc `secrets.token_urlsafe(16)`.

### 🟡 9.2. `st.rerun()` gọi sau khi append message rồi lại rerun

[src/ui/app.py:188](src/ui/app.py#L188): trong block xử lý câu hỏi đã hiển thị answer rồi vẫn `st.rerun()` ở cuối → render lại toàn bộ chat. Streamlit pattern này gây flicker. Có thể bỏ `st.rerun()` cuối và dựa vào auto-rerun của `chat_input`.

### 🟢 9.3. Pending question UX

[src/ui/app.py:144-146](src/ui/app.py#L144-L146): khi click example button, set `pending_question` rồi rerun. Pattern này OK nhưng nếu user click 2 button nhanh → race condition (pending bị overwrite). Edge case nhỏ.

---

## 10. Evaluation — `scripts/evaluate.py`

### 🔴 10.1. Mapping URL → article_id bằng slug có thể sai

[scripts/evaluate.py:91-95](scripts/evaluate.py#L91-L95):
```python
slug = url.rstrip("/").split("/")[-1]
article_ids.append(slug)
```

Nếu KG loader lưu `article_id = metadata.article_id` (vd `"win10-error-0x80070005"`) trong khi URL là `https://learn.microsoft.com/en-us/troubleshoot/windows-client/.../win10-error-0x80070005` thì slug khớp → OK. **Nhưng** nếu metadata từ scraper set article_id theo convention khác (hash, prefixed, có suffix `.json`), agent **luôn miss** dù tool đúng. Đây là nguyên nhân khả nghi cao của Hit@5 = 0.26 không tăng so với BM25.

**Fix:** verify bằng debug log:
```python
print(f"Agent URLs: {sources}")
print(f"Agent slugs: {article_ids}")
print(f"Gold article_id: {qa['article_id']}")
```
chạy 1 câu rồi check format.

### 🔴 10.2. Sample size n=50 quá nhỏ để claim significance

Hit@1: BM25=0.06 (3/50), Agent=0.18 (9/50). 95% Wilson CI:
- BM25: [0.013, 0.166]
- Agent: [0.097, 0.305]
**Hai khoảng overlap đáng kể.** Không claim được "Agent thắng có ý nghĩa thống kê".

**Fix:** tăng n ≥ 200. Chạy McNemar test trên matched pairs (cùng câu hỏi, BM25 hit vs Agent hit).

### 🟡 10.3. BM25 tokenization quá yếu

[scripts/evaluate.py:55-58](scripts/evaluate.py#L55-L58):
```python
corpus = [(a["title"] + " " + a["text"]).lower().split() for a in articles]
```
Chỉ split whitespace. Không strip dấu câu, không stopword, không stem. "fix" và "fixes" coi là 2 token khác. **Baseline yếu → "+200% improvement" bị thổi phồng.**

**Fix tối thiểu:**
```python
import re
corpus = [
    [t for t in re.findall(r"\w+", (a["title"] + " " + a["text"]).lower()) if len(t) > 2]
    for a in articles
]
```

### 🟡 10.4. Không đo answer quality

Cả pipeline xây để **trả lời**, eval chỉ đo retrieval (Hit@k, MRR). Field `answer` không được score.

**Fix:** thêm 1 metric:
- LLM-as-judge: đưa cả gold answer + agent answer cho LLM khác chấm 1–5
- BERTScore F1 với gold answer
- "Refusal rate": % câu agent nói "không có thông tin" (faithfulness)

### 🟡 10.5. Không có per-tool breakdown

Không biết tool nào contribute vào Hit@k. Có thể 90% lift đến từ EMBEDDING, BFS hầu như không hit. Cần:
```python
results_by_tool = defaultdict(list)
results_by_tool[response["tool_used"]].append(hit_at_k(...))
```

### 🟡 10.6. Single baseline (chỉ BM25)

Cần thêm:
- **KG-only baseline**: chạy `cypher_search(extract_entity(q))` không qua LLM router
- **Dense retrieval**: sentence-transformers + FAISS trên cùng article corpus
Tách được agent gain do KG hay do agent reasoning.

### 🟢 10.7. `time.sleep(2)` mỗi câu

50 × 2s = 100s overhead. Nếu Groq không rate-limit, bỏ. Nếu có rate-limit, dùng adaptive backoff.

### 🟢 10.8. Eval không reproducible nếu không có `data/raw/`

[scripts/evaluate.py:32-48](scripts/evaluate.py#L32-L48) load từ `data/raw/` (gitignored). Người clone repo về không chạy được eval. Nên cache BM25 corpus thành `.pkl` commit kèm.

### 🟢 10.9. Improvement % chia cho `max(bm25, 0.001)`

[scripts/evaluate.py:172](scripts/evaluate.py#L172): khi bm25 = 0.06, denominator = 0.06 → improvement = +200%. OK. Nhưng nếu bm25 = 0 thì = 0.001 → +18000% → vô nghĩa. Nên log absolute delta thay vì %.

---

## 11. Test Set Generation — `scripts/generate_testset.py`

### 🔴 11.1. Sample không phải "đều theo category"

[scripts/generate_testset.py:97-98](scripts/generate_testset.py#L97-L98):
```python
# Sample đều từ các categories
selected = all_files[:limit]
```
Comment nói "Sample đều", code thì cắt 50 file đầu sau khi `sorted()` → toàn category đầu alphabet (azure trước windows). Hoàn toàn bias.

**Fix:**
```python
import random
from collections import defaultdict

by_cat = defaultdict(list)
for f in all_files:
    by_cat[f.parent.name].append(f)

per_cat = limit // len(by_cat)
selected = []
for cat, files in by_cat.items():
    selected.extend(random.sample(files, min(per_cat, len(files))))
```

### 🔴 11.2. Test set "circular" — gold từ chính article generate ra câu hỏi

[scripts/generate_testset.py:24-46](scripts/generate_testset.py#L24-L46): LLM nhìn article X rồi sinh câu hỏi với `article_id = X`. **Không kiểm tra**:
- Câu hỏi có thực sự unique-answerable bởi article X? (Có thể article Y, Z cũng trả lời được)
- Câu hỏi có copy chữ từ article (vi phạm "Do NOT copy phrases")?

Hậu quả: nếu retrieval trả article Y đúng về mặt content nhưng `article_id != X` → tính là **miss** → underestimate cả BM25 và Agent.

**Fix:**
- Gold nên là **set** of article IDs (relevant articles), không phải single
- Cần human review subset (ít nhất 20%)

### 🟡 11.3. `text[:1500]` cắt ngẫu nhiên

[scripts/generate_testset.py:51](scripts/generate_testset.py#L51): chỉ thấy 1500 ký tự đầu → bias câu hỏi về phần intro của article.

### 🟡 11.4. `temperature=0.3` không cố định seed

[scripts/generate_testset.py:70](scripts/generate_testset.py#L70): test set không reproducible. Mỗi lần chạy ra câu hỏi khác nhau.

### 🟡 11.5. Không dùng GroqRotator

[scripts/generate_testset.py:88](scripts/generate_testset.py#L88) chỉ dùng `GROQ_API_KEY_1`. Nếu hết quota giữa chừng, fail im lặng → drop câu, test set không đủ 50.

### 🟡 11.6. `PROCESSED_DIR = Path("data/raw")` gây nhầm

[scripts/generate_testset.py:21](scripts/generate_testset.py#L21): biến tên `PROCESSED_DIR` nhưng trỏ vào `data/raw`. Reader sẽ confuse. Đổi tên thành `RAW_DIR`.

### 🟢 11.7. Không dedupe câu hỏi

Nếu 2 article rất giống (vd cùng version Windows update), LLM có thể sinh 2 câu hỏi gần như identical. Không có check trùng lặp.

---

## 12. Docker / Deployment

### 🔴 12.1. `docker-compose.yml` đang BROKEN

[docker-compose.yml:7-8, 25-26](docker-compose.yml):
```yaml
api:
  build:
    context: .
    dockerfile: Dockerfile      # <-- file này không tồn tại
ui:
  build:
    context: .
    dockerfile: Dockerfile      # <-- cũng vậy
```

Repo có `Dockerfile.api` và `Dockerfile.ui` (sau commit `8407ae1`) nhưng compose chưa update. **`docker-compose up --build` sẽ fail.**

**Fix:**
```yaml
api:
  build:
    context: .
    dockerfile: Dockerfile.api
ui:
  build:
    context: .
    dockerfile: Dockerfile.ui
```

### 🟡 12.2. Cả hai Dockerfile cài full requirements.txt

`Dockerfile.ui` cài cả torch, torch-geometric, neo4j driver (qua requirements.txt) nhưng UI chỉ dùng `streamlit + requests`. Container UI nặng vài GB không cần thiết.

**Fix:** tách `requirements-api.txt` và `requirements-ui.txt`.

### 🟡 12.3. `Dockerfile.api` copy `data/communities.json` và `community_summaries.json` riêng lẻ

[Dockerfile.api:8-9](Dockerfile.api#L8-L9): hardcode 2 file. Nếu thêm `data/test_set.json` (cần cho eval container), phải edit Dockerfile. Nên `COPY data/*.json ./data/`.

### 🟡 12.4. Healthcheck dùng `curl`

[docker-compose.yml:17](docker-compose.yml#L17): `curl` được cài trong Dockerfile. OK. Nhưng có thể dùng `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"` để không cần apt-get curl → image nhỏ hơn.

### 🟡 12.5. Logs volume mount chỉ cho api

[docker-compose.yml:14-15](docker-compose.yml#L14-L15): chỉ api mount `./logs`. UI cũng có log từ `requests` errors mà không persist.

### 🟢 12.6. Không có `.dockerignore` đầy đủ

`.dockerignore` tồn tại nhưng chưa đọc — kiểm tra có ignore `data/raw/`, `data/processed/`, `__pycache__`, `.git`, `venv/`, `models/gcn_checkpoint/` (training artifact, không cần cho serve)?

---

## 13. Code Quality / Hygiene

### 🟡 13.1. `requirements.txt` lưu UTF-16

Đọc file thấy mỗi ký tự xen khoảng trắng → encoding UTF-16 LE BOM. Trên Linux/Mac, `pip install -r requirements.txt` có thể parse fail. Convert sang UTF-8.

### 🟡 13.2. Thiếu nhiều dependency thực tế trong `requirements.txt`

Đã nêu ở [§6.6](#-66-sentence-transformers-không-có-trong-requirementstxt). Tổng kết:
- `torch`
- `torch-geometric` (cùng `torch-scatter`, `torch-sparse` cho 1 số version)
- `sentence-transformers`
- `graspologic`
- `tavily-python`
- `rank-bm25`
- `numpy` (transitive nhưng nên explicit)

### 🟡 13.3. Mixed Vietnamese/English trong code

Comments + log message tiếng Việt (`logger.info("Kết nối Neo4j thành công")`) xen với English. Nhất quán tiếng Việt hoặc English — không phải vấn đề kỹ thuật, nhưng nếu mở source ra thế giới thì nên English.

### 🟡 13.4. `tests/` hoàn toàn rỗng

Chỉ có `__init__.py`. Không có 1 unit test nào cho:
- `cypher_search` happy path / empty entity / không tồn tại
- `embedding_search` threshold
- `is_empty_result` cho mỗi tool
- `format_context` không crash với data lạ

Khuyến nghị: ít nhất 5–10 test cho `agent.py` core logic. Dùng `pytest` + mock Groq/Neo4j.

### 🟡 13.5. Không có CI

Không có `.github/workflows/`. PR mở ra không có check tự động. Tối thiểu nên có:
- `pip install -r requirements.txt` (verify dependencies sync)
- `python -c "from src.agent.agent import ITHelpdeskAgent"` (smoke import)
- Linter (ruff/flake8)

### 🟢 13.6. Logging level mặc định INFO

OK, nhưng không có log rotation. `logs/agent.log` có thể grow indefinitely. Thêm `RotatingFileHandler`.

### 🟢 13.7. Magic numbers rải rác

- `top_k=5` ở `embedding_search`
- `LIMIT 20`, `LIMIT 5` ở Cypher
- `[*..6]` BFS depth
- `text[:2000]`, `text[:1500]`
- `temperature=0`, `max_tokens=512`
- `len(history) > 20`

Gom thành `src/config.py` hoặc `configs/agent.yaml` để dễ tune.

---

## 14. Documentation / README

### 🟡 14.1. README claim không chính xác

[readme.md:16](readme.md#L16): "ReAct Agent (4 tools)" — nhưng [§1.1](#-11-không-phải-agentic-ai-thực-sự) đã chứng minh không phải ReAct.

### 🟡 14.2. README còn nhắc "Dockerfile" (singular)

[readme.md:221](readme.md#L221) trong tree: `├── Dockerfile`. Sau khi đã tách 2 file, README chưa update.

### 🟡 14.3. README mention `cleaner.py` không tồn tại

[readme.md:196](readme.md#L196): `├── ingestion/          # scraper.py, cleaner.py`. Repo không có `cleaner.py`. Hoặc xóa khỏi README, hoặc thực sự tách `cleaner.py` ra.

### 🟡 14.4. README không nhắc các config bắt buộc khác

`.env.example` được nhắc nhưng không có file đó trong repo (đã check Glob trước đó). User clone về không có template → phải tự đoán biến nào cần.

**Fix:** tạo `.env.example` với:
```env
GROQ_API_KEY_1=
GROQ_API_KEY_2=
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=
TAVILY_API_KEY=
```

### 🟡 14.5. Setup docs thiếu các bước phụ thuộc

README không nhắc:
- Phải tạo Neo4j AuraDB account thế nào
- Database trống ban đầu, phải chạy `kg_loader` xong mới `train_gcn`
- Sau khi train_gcn xong, model file phải có ở local trước khi chạy api (Dockerfile copy `models/`)

### 🟢 14.6. Bảng kết quả eval lệch giữa README và `eval_results.json`

[readme.md:181-187](readme.md#L181-L187):
```
| Hit@1  | 0.060 | 0.180 | +200.0%
| Hit@5  | 0.260 | 0.260 | +0.0%
| MRR    | 0.115 | 0.217 | +87.9%
```
Khớp với `data/eval_results.json` ✓. Nhưng README không nói rõ test set chỉ có 50 câu (số nhỏ → CI overlap, không claim significance được).

---

## 15. Tổng hợp ưu tiên fix

### 🔴 PHẢI sửa (~1–2 buổi)

1. [§12.1](#-121-docker-composeyml-đang-broken) — Fix `docker-compose.yml` trỏ vào đúng `Dockerfile.api`/`Dockerfile.ui`
2. [§3.3](#-33-thiếu-hoàn-toàn-indexconstraint) + [§3.4](#-34-driver-không-phải-singleton--mỗi-query-tạo-connection-mới) — Thêm CONSTRAINT/INDEX vào `kg_loader`, làm Neo4j driver singleton
3. [§3.1](#-31-bug-where-chỉ-check-entity-ở-slot-e-bỏ-qua-e2) + [§3.2](#-32-bug-empty-entity_name--match-mọi-entity) — Fix WHERE bug, guard empty entity
4. [§4.1](#-41-cypher-injection-nhẹ-ở-relation-type) — Whitelist relation type trước khi f-string
5. [§7.1](#-71-format-output-không-khớp-với-format-agent-đọc) — Unify schema giữa `community.py` và `summarizer.py`
6. [§8.1](#-81-memory-leak-sessions-tích-lũy-vô-hạn) — TTL cache cho FastAPI sessions, hoặc DELETE thực sự xóa
7. [§10.1](#-101-mapping-url--article_id-bằng-slug-có-thể-sai) — Verify mapping URL → article_id, có thể lý do Hit@5 không tăng
8. [§11.1](#-111-sample-không-phải-đều-theo-category) + [§11.2](#-112-test-set-circular--gold-từ-chính-article-generate-ra-câu-hỏi) — Random/stratified sample test set, gold thành multi-relevant
9. [§6.1](#-61-validation-dùng-train_datax-thay-vì-val_datax) — Fix GAE link prediction validation
10. [§6.6](#-66-sentence-transformers-không-có-trong-requirementstxt) + [§13.2](#-132-thiếu-nhiều-dependency-thực-tế-trong-requirementstxt) — Bổ sung dependencies vào requirements

### 🟡 NÊN sửa (~3–5 buổi)

11. [§2.1](#-21-bug-display_entity-dùng-dir-để-check-biến-local) – [§2.6](#-26-llm_call-âm-thầm-trả--khi-lỗi) — Loạt bug nhỏ trong agent core
12. [§3.5](#-35-limit-không-có-order-by--kết-quả-non-deterministic) – [§3.9](#-39-bfs_search-không-kiểm-name1--name2) — Cải thiện CYPHER/BFS query
13. [§4.4](#-44-loader-không-idempotent-với-entity-rename) – [§4.6](#-46-một-file--một-transaction-implicit-không-atomic) — Loader robustness
14. [§5.1](#-51-text2000-cắt-giữa-chừng) – [§5.4](#-54-timesleep2-hard-coded-sau-mỗi-article) — Entity extraction quality
15. [§10.2](#-102-sample-size-n50-quá-nhỏ-để-claim-significance) – [§10.6](#-106-single-baseline-chỉ-bm25) — Eval rigor (n≥200, McNemar, answer quality, multiple baselines)
16. [§11.3](#-113-text1500-cắt-ngẫu-nhiên) – [§11.6](#-116-processed_dir--pathdataraw-gây-nhầm) — Test set generation quality
17. [§12.2](#-122-cả-hai-dockerfile-cài-full-requirementstxt) – [§12.5](#-125-logs-volume-mount-chỉ-cho-api) — Docker hygiene

### 🟢 NICE TO HAVE (~tuần+)

18. [§1.1](#-11-không-phải-agentic-ai-thực-sự) — Refactor agent thành ReAct loop thật
19. [§13.4](#-134-tests-hoàn-toàn-rỗng) – [§13.5](#-135-không-có-ci) — Test suite + CI
20. [§14.x](#14-documentation--readme) — README cleanup, `.env.example`, đồng bộ tree

---

## Phụ lục A — Smoke test sau khi fix

Chạy theo thứ tự để verify:

```bash
# 1. Verify deps
pip install -r requirements.txt
python -c "from src.agent.agent import ITHelpdeskAgent; print('imports ok')"

# 2. Verify Neo4j có index
python -c "
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()
d = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'),os.getenv('NEO4J_PASSWORD')))
with d.session() as s:
    print(list(s.run('SHOW INDEXES')))
"

# 3. Verify CYPHER không bị empty entity bug
python -c "
from src.agent.neo4j_query import cypher_search
r = cypher_search('')
assert r['relations'] == [] and r['articles'] == [], 'EMPTY ENTITY BUG'
print('cypher empty guard ok')
"

# 4. Verify Docker build
docker-compose build

# 5. Verify mapping URL → article_id
python -c "
import json
test = json.load(open('data/test_set.json'))
print('Sample article_id:', test[0]['article_id'])
# So sánh với slug từ URL của article tương ứng
"
```

## Phụ lục B — Nguồn tham khảo

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022) — định nghĩa ReAct loop
- Neo4j docs — fulltext index: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/
- PyG GAE example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
- McNemar test cho retrieval comparison: scipy.stats.contingency.mcnemar
