# 📈 Chiến lược Học liên tục (Continual Learning) — IT Helpdesk KBQA

> **Hệ thống:** Hỏi đáp dựa trên Đồ thị Tri thức cho IT Helpdesk  
> **Phạm vi:** Thiết kế khái niệm cho sự tiến hóa mô hình, giám sát và giảm thiểu trôi dạt

---

## 1. Tổng quan

Hệ thống IT Helpdesk KBQA bao gồm **5 thành phần có thể huấn luyện/tiến hóa**, mỗi thành phần có chu kỳ cập nhật và rủi ro trôi dạt khác nhau:

| # | Thành phần | Loại | Trạng thái hiện tại | Tần suất cập nhật |
|---|---|---|---|---|
| 1 | **Đồ thị Tri thức** (Neo4j) | Kho dữ liệu | 3.266 thực thể, 7.000+ quan hệ | Hàng tháng |
| 2 | **GCN Embeddings** (GAE) | Mô hình neural | Vector 32 chiều cho mỗi node | Sau khi KG cập nhật |
| 3 | **Retriever đã finetune** | Mô hình neural | msmarco-MiniLM-L6 đã finetune | Hàng quý |
| 4 | **LLM** (Groq LLaMA 3.1) | API bên ngoài | Quản lý bởi Groq | Do nhà cung cấp |
| 5 | **Tóm tắt cộng đồng** | Văn bản sinh bởi LLM | Leiden + LLM summaries | Sau khi KG cập nhật |

```
Nguồn dữ liệu ──► Đồ thị Tri thức ──► GCN Embeddings ──► Agent Runtime
      ↑                  ↑                    ↑                  ↑
  Bài viết mới      Thay đổi schema    Huấn luyện lại      Tinh chỉnh prompt
  Danh mục mới      Quan hệ mới        Feature mới          Tool mới
```

---

## 2. Chiến lược Thu thập Dữ liệu mới

### 2.1 Pipeline dữ liệu tự động

```
┌─────────────────────────────────────────────────────────────┐
│                VÒNG LẶP THU THẬP DỮ LIỆU                    │
│                                                              │
│  ┌──────────────────┐   Lên lịch    ┌──────────────────────┐│
│  │ Microsoft Learn   │──(hàng tháng)─►│ discover_urls.py    ││
│  │ ToC API           │               │ (khám phá URL mới)   ││
│  └──────────────────┘               └────────┬─────────────┘│
│                                              │              │
│                                So sánh với URL đã có        │
│                                              │              │
│  ┌──────────────────┐               ┌────────▼─────────────┐│
│  │ Bài viết         │◄──────────────│ scraper.py           ││
│  │ mới/cập nhật     │               │ (chỉ scrape URL mới) ││
│  └──────┬───────────┘               └──────────────────────┘│
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────┐   ┌────────────────┐                  │
│  │ entity_           │──►│ kg_loader.py   │──► Neo4j         │
│  │ extractor.py      │   │ (nạp tăng dần) │   (MERGE nodes)  │
│  └──────────────────┘   └────────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Các nguồn thu thập:**

| Nguồn | Phương pháp | Tần suất | Loại dữ liệu |
|---|---|---|---|
| Microsoft Learn ToC API | Crawl tự động file JSON mục lục | Hàng tháng | Bài troubleshooting mới |
| Log truy vấn người dùng | Ghi log từ FastAPI | Liên tục | Câu hỏi chưa có câu trả lời tốt |
| Log thất bại của Agent | Observation = "No results found" | Liên tục | Lỗ hổng tri thức |
| Phản hồi người dùng | Nút đánh giá (👍/👎) trên UI | Mỗi truy vấn | Tín hiệu chất lượng câu trả lời |

### 2.2 Phát hiện Lỗ hổng Tri thức

```
Agent ghi log "No results found" cho thực thể X
    ↓ tổng hợp hàng tuần
    ↓ phân cụm theo chủ đề
    ↓
"Top 10 chủ đề chưa được phủ trong tuần:
  1. Windows 11 24H2 BitLocker (17 truy vấn)
  2. Lỗi tích hợp Copilot (12 truy vấn)
  3. ..."
    ↓
Kích hoạt scraping có mục tiêu cho các chủ đề này
```

---

## 3. Chiến lược Huấn luyện lại & Tinh chỉnh

### 3.1 Ma trận cập nhật theo thành phần

| Thành phần | Điều kiện kích hoạt | Phương pháp | Thời gian ngưng | Rollback |
|---|---|---|---|---|
| **Đồ thị Tri thức** | Có bài viết mới được scrape | Nạp tăng dần bằng `MERGE` (không xóa) | Không có | Xóa node mới theo batch_id |
| **GCN Embeddings** | Cấu trúc KG thay đổi >5% | Huấn luyện lại toàn bộ (train_gcn.py) | ~5 phút | Giữ checkpoint trước đó |
| **Retriever** | Có >200 cặp finetune mới | Tiếp tục finetune từ checkpoint cuối | ~10 phút | Quay về model trước |
| **Phát hiện cộng đồng** | KG thay đổi >10% | Chạy lại thuật toán Leiden | ~2 phút | Giữ file communities.json cũ |
| **Tóm tắt cộng đồng** | Cộng đồng thay đổi | Chạy lại summarizer.py | ~5 phút | Giữ file summaries cũ |

### 3.2 Cập nhật Đồ thị Tri thức (Hàng tháng)

```
Giai đoạn 1: Khám phá
  python scripts/discover_urls.py
  → So sánh với discovered_urls.json hiện có
  → Xác định các URL MỚI

Giai đoạn 2: Scrape tăng dần
  python -m src.ingestion.scraper --urls data/new_urls.json
  → Chỉ scrape bài viết mới

Giai đoạn 3: Trích xuất thực thể
  python -m src.kg_build.entity_extractor --input data/raw/new/
  → Trích xuất thực thể chỉ từ bài mới

Giai đoạn 4: Nạp tăng dần
  python -m src.kg_build.kg_loader --input data/processed/new/
  → MERGE vào đồ thị hiện có (chỉ thêm, không xóa)
  → Gắn tag batch_id cho node mới để có thể rollback

Giai đoạn 5: Kiểm chứng
  → Đếm node/relation mới
  → Kiểm tra node cô lập (orphan)
  → Xác minh tính toàn vẹn ràng buộc
```

### 3.3 Huấn luyện lại GCN Embeddings

**Điều kiện:** Khi cấu trúc KG thay đổi >5% (node/edge mới so với snapshot lần huấn luyện trước).

```
Bước 1: Xuất đồ thị cập nhật từ Neo4j
Bước 2: Huấn luyện lại GAE với CÙNG siêu tham số
         - GCN 2 lớp (input→64→32)
         - Early stopping trên validation loss
         - Giữ model trước để dự phòng
Bước 3: So sánh embeddings mới vs cũ
         - Phân phối cosine similarity
         - Nếu trung bình lệch > 0.3 → cảnh báo, review thủ công
Bước 4: Hoán đổi nóng file embeddings
         - Thay thế models/embeddings/node_embeddings.npy
         - API tự nhận khi khởi động lại
```

### 3.4 Tinh chỉnh Retriever (Hàng quý)

```
Bước 1: Sinh cặp huấn luyện mới
         python scripts/generate_finetune_data.py
         → Bao gồm thực thể KG mới

Bước 2: Tiếp tục finetune (KHÔNG từ đầu)
         - Load từ models/retriever/ (checkpoint cuối)
         - Huấn luyện trên dữ liệu cũ + mới kết hợp
         - Ít epoch hơn (3-5 so với 10 ban đầu)

Bước 3: Đánh giá trên tập test riêng
         - Nếu MRR giảm > 10% → từ chối, điều tra
         - Nếu MRR cải thiện hoặc ổn định → triển khai

Bước 4: Cập nhật đặc trưng đầu vào GCN
         - Mã hóa lại tên node bằng retriever mới
         - Huấn luyện lại GCN (Bước 3.3)
```

---

## 4. Phát hiện Suy giảm Hiệu suất

### 4.1 Các chỉ số Giám sát

#### Tầng 1 — Giám sát thời gian thực (mỗi truy vấn)

| Chỉ số | Nguồn | Ngưỡng cảnh báo | Phát hiện điều gì |
|---|---|---|---|
| **Tỷ lệ observation rỗng** | Log agent | >40% truy vấn | Lỗ hổng phủ của KG |
| **Tỷ lệ fallback WEBSEARCH** | Log agent | >50% truy vấn | KG đang lỗi thời |
| **Số bước trung bình mỗi truy vấn** | Log agent | >3.5 trung bình | Agent gặp khó khăn tìm câu trả lời |
| **Độ trễ phản hồi p95** | Log API | >15 giây | Suy giảm hiệu suất |
| **Tỷ lệ lỗi tool** | Log agent | >5% | Tính khả dụng Neo4j/Tavily |

#### Tầng 2 — Đánh giá theo lô hàng tuần

| Chỉ số | Phương pháp | Ngưỡng cảnh báo | Phát hiện điều gì |
|---|---|---|---|
| **Hit@1 trên tập test** | Chạy evaluate.py hàng tuần | Giảm >15% so với baseline | Suy giảm chất lượng truy xuất |
| **MRR trên tập test** | Chạy evaluate.py hàng tuần | Giảm >10% so với baseline | Chất lượng xếp hạng tổng thể |
| **ROUGE-L trên tập test** | Chạy evaluate.py hàng tuần | Giảm >20% so với baseline | Chất lượng sinh câu trả lời |
| **Độ phủ thực thể** | Đếm thực thể duy nhất trong truy vấn vs KG | Phủ <60% | Lỗ hổng tri thức KG |

#### Tầng 3 — Đánh giá chiến lược hàng tháng

| Chỉ số | Phương pháp | Hành động |
|---|---|---|
| **Mức hài lòng người dùng** | Nút đánh giá trên UI (👍/👎) | Xem lại câu trả lời bị đánh giá thấp |
| **Dịch chuyển phân phối chủ đề** | So sánh chủ đề truy vấn giữa các tháng | Xác định vấn đề IT mới nổi |
| **Điểm độ mới của KG** | Tuổi trung bình bài viết trong KG | Kích hoạt re-scrape nếu >6 tháng |
| **Kiểm tra phiên bản LLM** | So sánh phiên bản model Groq | Cập nhật prompt nếu model thay đổi |

### 4.2 Pipeline Cảnh báo Tự động

```
┌─────────────────────────────────────────────────┐
│             PIPELINE GIÁM SÁT                    │
│                                                  │
│  Log Agent ──► Tổng hợp chỉ số (mỗi giờ)        │
│       │              │                           │
│       │              ├── Tỷ lệ obs rỗng > 40%    │
│       │              ├── Tỷ lệ fallback > 50%    │
│       │              ├── Độ trễ p95 > 15s         │
│       │              └── Tỷ lệ lỗi > 5%          │
│       │                     │                    │
│       │              ┌──────▼──────┐             │
│       │              │  CẢNH BÁO?  │             │
│       │              └──────┬──────┘             │
│       │                     │ CÓ                 │
│       │              ┌──────▼──────┐             │
│       │              │ Thông báo   │             │
│       │              │ + ghi log   │             │
│       │              └─────────────┘             │
│       │                                          │
│       ▼                                          │
│  Hàng tuần: Chạy evaluate.py trên tập test      │
│  Hàng tháng: Bảng điều khiển đánh giá chiến lược │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 5. Rủi ro Trôi dạt Mô hình & Chiến lược Giảm thiểu

### 5.1 Ma trận Rủi ro

| Rủi ro | Khả năng xảy ra | Mức ảnh hưởng | Thành phần bị ảnh hưởng |
|---|---|---|---|
| **Trôi dạt dữ liệu — Chủ đề IT mới** | 🔴 Cao | 🟡 Trung bình | KG, Retriever |
| **Trôi dạt dữ liệu — Giải pháp lỗi thời** | 🟡 Trung bình | 🔴 Cao | KG (bản sửa lỗi cũ) |
| **Trôi dạt khái niệm — LLM thay đổi hành vi** | 🟡 Trung bình | 🟡 Trung bình | Agent (prompt có thể hỏng) |
| **Trôi dạt thượng nguồn — Groq cập nhật model** | 🟢 Thấp | 🔴 Cao | Tất cả thành phần dùng LLM |
| **Trôi dạt schema — Loại thực thể mới** | 🟢 Thấp | 🟡 Trung bình | KG, Entity Extractor |
| **Trôi dạt embedding — Retriever suy giảm** | 🟡 Trung bình | 🟡 Trung bình | Retriever, GCN |

### 5.2 Phân tích Rủi ro Chi tiết

#### Rủi ro 1: Trôi dạt dữ liệu — Chủ đề IT mới (Khả năng CAO)

**Vấn đề:** Microsoft phát hành sản phẩm/tính năng mới (ví dụ: Copilot, Windows 12) → người dùng đặt câu hỏi → KG không có thực thể liên quan → agent phải fallback sang tìm kiếm web.

**Tín hiệu phát hiện:**
- Tỷ lệ fallback WEBSEARCH tăng theo tháng
- Tên thực thể mới xuất hiện trong truy vấn nhưng không có trong KG
- Tỷ lệ "No results found" tăng cho danh mục cụ thể

**Chiến lược giảm thiểu:**
```
1. Khám phá URL tự động hàng tháng (phát hiện bài viết mới trên Microsoft Learn)
2. Phân tích lỗ hổng: phân cụm truy vấn thất bại → xác định chủ đề thiếu
3. Scraping có mục tiêu + cập nhật KG tăng dần
4. Đánh giá lại sau cập nhật để xác nhận cải thiện độ phủ
```

#### Rủi ro 2: Trôi dạt dữ liệu — Giải pháp lỗi thời (Khả năng TRUNG BÌNH)

**Vấn đề:** Các bản sửa lỗi cũ trở nên không hợp lệ (ví dụ: cách sửa Windows 10 được áp dụng cho câu hỏi Windows 11). KG chứa thông tin lỗi thời → agent đưa ra lời khuyên đã cũ.

**Tín hiệu phát hiện:**
- Đánh giá 👎 từ người dùng về các chủ đề cụ thể
- Kết quả tìm kiếm web mâu thuẫn với câu trả lời từ KG
- Bài viết cũ hơn 12 tháng cho các chủ đề biến đổi nhanh

**Chiến lược giảm thiểu:**
```
1. Gắn tag scrape_date trong metadata bài viết
2. Tính điểm độ mới: giảm điểm thực thể từ bài viết cũ
3. Ưu tiên WEBSEARCH cho truy vấn liên quan phiên bản cụ thể
   (đã triển khai một phần qua regex pre-routing)
4. Re-scrape định kỳ các URL hiện có để phát hiện cập nhật nội dung
```

#### Rủi ro 3: Trôi dạt khái niệm — LLM thay đổi hành vi (Khả năng TRUNG BÌNH)

**Vấn đề:** Groq cập nhật trọng số hoặc lượng tử hóa model LLaMA → prompt của agent có thể tạo ra hành vi tool-calling khác → chất lượng thay đổi một cách âm thầm.

**Tín hiệu phát hiện:**
- Phân phối tool dịch chuyển (ví dụ: đột nhiên 80% EMBEDDING, 5% CYPHER)
- Độ dài/phong cách câu trả lời thay đổi
- Hit@1/MRR giảm trên tập test ổn định

**Chiến lược giảm thiểu:**
```
1. Ghim phiên bản model cụ thể trong code (ví dụ: "llama-3.1-8b-instant")
2. Đánh giá khóa phiên bản: chạy tập test sau mỗi thông báo của Groq
3. Test hồi quy prompt: 10 câu hỏi chuẩn với tool + entity kỳ vọng
4. Nếu chất lượng giảm → cập nhật prompt phù hợp hành vi model mới
```

#### Rủi ro 4: Trôi dạt thượng nguồn — API Groq thay đổi (Khả năng THẤP, Ảnh hưởng CAO)

**Vấn đề:** Groq ngừng hỗ trợ model, thay đổi schema API, hoặc điều chỉnh rate limit.

**Chiến lược giảm thiểu:**
```
1. Groq SDK dùng format tương thích OpenAI → dễ chuyển nhà cung cấp
2. Duy trì cấu hình nhà cung cấp dự phòng (ví dụ: Together AI, OpenRouter)
3. Trừu tượng hóa lời gọi LLM qua interface (đã có một phần qua llm_call())
4. Theo dõi trang trạng thái / changelog của Groq
```

#### Rủi ro 5: Trôi dạt Embedding — Retriever suy giảm (Khả năng TRUNG BÌNH)

**Vấn đề:** Khi KG mở rộng, phân phối huấn luyện của retriever lệch khỏi phân phối thực thể thực tế → độ chính xác khớp thực thể giảm.

**Tín hiệu phát hiện:**
- Tỷ lệ fallback fuzzy match tăng (retriever trả về kết quả confidence thấp)
- Embedding search trả về thực thể không liên quan
- Hit@K suy giảm cụ thể cho các thực thể mới

**Chiến lược giảm thiểu:**
```
1. Finetune lại retriever hàng quý với thực thể KG cập nhật
2. Giám sát điểm semantic match (ghi log vi phạm ngưỡng)
3. So sánh A/B: retriever cũ vs mới trên cùng truy vấn
4. Chuỗi fallback: retriever → fuzzy match → yêu cầu người dùng làm rõ
```

---

## 6. Vòng đời Học liên tục

```
         ┌──────────────────────────────────────────────┐
         │                                              │
         ▼                                              │
  ┌─────────────┐    ┌─────────────┐    ┌────────────┐ │
  │  THU THẬP   │───►│  CẬP NHẬT   │───►│  ĐÁNH GIÁ  │ │
  │             │    │             │    │            │ │
  │ • URL mới   │    │ • Scrape    │    │ • Tập test │ │
  │ • Log user  │    │ • Trích xuất│    │ • Chỉ số   │ │
  │ • Phát hiện │    │ • Nạp KG    │    │ • So sánh  │ │
  │   lỗ hổng   │    │ • Huấn luyện│    └──────┬─────┘ │
  └─────────────┘    └─────────────┘           │       │
                                               ▼       │
                                        ┌────────────┐ │
                                        │ TRIỂN KHAI? │ │
                                        │            │ │
                                        │ Chất lượng │ │
                                        │ OK → Deploy│ │
                                        │ Giảm       │ │
                                        │ → Rollback │ │
                                        └──────┬─────┘ │
                                               │       │
                                               ▼       │
                                        ┌────────────┐ │
                                        │  GIÁM SÁT  │─┘
                                        │            │
                                        │ • Cảnh báo │
                                        │ • Trôi dạt │
                                        │ • Phản hồi │
                                        └────────────┘

  Dòng thời gian:
  ─────────────────────────────────────────────────────►
  Liên tục        Hàng tháng     Hàng quý       Hàng năm
  (ghi log)       (cập nhật KG)  (retriever)    (đánh giá kiến trúc)
```

---

## 7. Bảng tóm tắt Deliverables

| Deliverable | Trạng thái | Vị trí |
|---|---|---|
| ✅ Cách thu thập dữ liệu mới | Hoàn thành | §2 — Pipeline tự động + phát hiện lỗ hổng |
| ✅ Cách huấn luyện lại / tinh chỉnh | Hoàn thành | §3 — Ma trận cập nhật theo thành phần |
| ✅ Cách phát hiện suy giảm hiệu suất | Hoàn thành | §4 — Hệ thống giám sát 3 tầng |
| ✅ Các chỉ số giám sát đề xuất | Hoàn thành | §4.1 — 14 chỉ số trên 3 tầng |
| ✅ Rủi ro trôi dạt mô hình | Hoàn thành | §5.1 — 6 rủi ro với ma trận Khả năng × Ảnh hưởng |
| ✅ Chiến lược giảm thiểu | Hoàn thành | §5.2 — Kế hoạch giảm thiểu chi tiết cho từng rủi ro |
