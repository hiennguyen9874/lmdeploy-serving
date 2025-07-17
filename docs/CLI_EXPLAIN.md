Dưới đây là phần **giải thích chi tiết, kỹ lưỡng** về từng thông số trong cấu hình bạn đang sử dụng với `lmdeploy serve` để triển khai **mô hình phân loại ảnh sử dụng LLM (Vision + LLM)**. Mỗi thông số được phân tích theo:

- ✅ **Tác dụng**
- 📊 **Ảnh hưởng tới hiệu suất / bộ nhớ / độ trễ**
- 🔧 **Gợi ý điều chỉnh (nếu cần)**

---

## ⚙️ Tham số chi tiết

---

### `--backend turbomind`

✅ **Tác dụng**: Chỉ định backend inference là `TurboMind` (thay vì `pytorch`).

📊 **Ưu điểm**:

- Nhanh hơn đáng kể trong môi trường production (sử dụng CUDA kernel tối ưu).
- Dùng CUDA Graph, multi-stream GPU và tối ưu memory allocation.

🔧 **Gợi ý**:

- Nên dùng nếu bạn phục vụ real-time, đồng thời cao.
- Chỉ cần dùng `pytorch` khi bạn cần huấn luyện hoặc debug thấp.

---

### `--max-concurrent-requests 64`

✅ **Tác dụng**: Giới hạn số request đồng thời mà FastAPI server xử lý trước khi đưa vào hàng đợi của engine.

📊 **Ảnh hưởng**:

- Tăng giá trị giúp xử lý nhiều request hơn trong cùng thời điểm.
- Quá cao có thể gây nghẽn GPU hoặc lỗi OOM nếu batch quá lớn hoặc tài nguyên không đủ.

🔧 **Gợi ý**:

- Bắt đầu với 32–64 nếu bạn có GPU \~24GB.
- Kiểm tra bằng `nvidia-smi` để điều chỉnh cho phù hợp.

---

### `--disable-fastapi-docs`

✅ **Tác dụng**: Vô hiệu hóa Swagger UI, ReDoc và OpenAPI schema tự động.

📊 **Ảnh hưởng**:

- Tránh tạo route `/docs`, `/redoc`, `/openapi.json`, giảm overhead nhẹ.
- Giảm nguy cơ lộ API docs trong môi trường production.

🔧 **Gợi ý**:

- Luôn bật trong môi trường production.
- Trong dev có thể bật để test API.

---

### `--log-level ERROR`

✅ **Tác dụng**: Giới hạn log chỉ hiển thị lỗi (`ERROR`) trở lên.

📊 **Ảnh hưởng**:

- Tăng hiệu suất vì không phải xử lý log `INFO`, `DEBUG`.
- Tránh spam log khi có nhiều request đồng thời.

🔧 **Gợi ý**:

- Nếu bạn cần profile hiệu năng → có thể tạm chuyển sang `INFO`.

---

### `--session-len 64`

✅ **Tác dụng**: Số token tối đa được giữ trong một phiên (bao gồm prompt và output).

📊 **Ảnh hưởng**:

- Token vượt quá sẽ gây lỗi "run out of tokens".
- Giá trị nhỏ phù hợp với các truy vấn chỉ hỏi–đáp một câu (image → label).

🔧 **Gợi ý**:

- Có thể tăng lên `128` hoặc `256` nếu prompt/model cần nhiều token hơn.
- Với vision model có phần embedding ảnh → cần cao hơn chút (`512` an toàn).

---

### `--max-batch-size 32`

✅ **Tác dụng**: Số lượng request tối đa được gom vào một batch xử lý một lượt (để tiết kiệm GPU round-trip).

📊 **Ảnh hưởng**:

- Tăng throughput rất nhiều nếu GPU đủ mạnh.
- Nhưng tăng độ trễ nếu batch bị chờ đầy quá lâu.

🔧 **Gợi ý**:

- Dùng `16–64` tùy GPU (với A100 có thể đẩy tới 64–128).
- Kiểm tra thời gian phản hồi nếu batch quá lớn.

---

### `--cache-max-entry-count 0.2`

✅ **Tác dụng**: Tỷ lệ GPU memory dành cho cache attention key/value.

📊 **Ảnh hưởng**:

- Nếu bạn dùng chat session, cache giúp giảm tính toán khi lặp lại prompt cũ.
- Nhưng vì bạn **không dùng chat**, nên nên **giảm xuống 0.1 hoặc 0.2** để giải phóng GPU cho request mới.

🔧 **Gợi ý**:

- Dùng `0.1–0.2` cho request độc lập.
- Dùng `0.8` nếu cần cache prompt dài/đa lượt.

---

### `--cache-block-seq-len 64`

✅ **Tác dụng**: Số token trên mỗi block attention cache.

📊 **Ảnh hưởng**:

- Phù hợp cho prompt và response ngắn (classification image to label).
- Nếu prompt dài hơn 64 token → nhiều block hơn → tốn RAM hơn.

🔧 **Gợi ý**:

- Để `64` nếu mô hình dùng kiểu prompt: "What is in this image? \[IMAGE]".
- Tăng lên `128–256` nếu prompt dài.

---

### `--max-prefill-token-num 128`

✅ **Tác dụng**: Số lượng token tối đa model sẽ xử lý trong prefill stage (prompt đầu vào).

📊 **Ảnh hưởng**:

- Tối ưu cho prompt ngắn.
- Nếu prompt + vision embedding vượt quá, sẽ bị lỗi.

🔧 **Gợi ý**:

- Tăng lên `256` nếu bạn dùng image captioning + prompt template.
- Giữ `128` nếu chỉ cần classification label.

---

### `--quant-policy 4`

✅ **Tác dụng**: Dùng quantization 4-bit cho attention key/value trong quá trình inference.

📊 **Ảnh hưởng**:

- Giảm tiêu thụ VRAM \~50%.
- Tăng throughput (vì nhỏ hơn nên xử lý nhanh).
- Có thể giảm một chút độ chính xác, nhưng thường không đáng kể với classification.

🔧 **Gợi ý**:

- Dùng `4` nếu bạn đang chạy mô hình >7B trên GPU <48GB.
- `0` nếu bạn cần độ chính xác tuyệt đối hoặc đang benchmark.

---

### `--vision-max-batch-size 32`

✅ **Tác dụng**: Số lượng ảnh tối đa mô hình xử lý mỗi batch.

📊 **Ảnh hưởng**:

- Ảnh càng nhiều, throughput càng cao nhưng cần nhiều GPU VRAM.
- Đặt quá cao có thể gây OOM (out-of-memory).

🔧 **Gợi ý**:

- Dùng `16–32` nếu ảnh đầu vào khoảng 224x224 – 512x512.
- Giảm xuống `8` nếu ảnh lớn hoặc GPU nhỏ.

---

## ✅ Tổng kết cấu hình đề xuất (cho môi trường production, phân loại ảnh)

```bash
lmdeploy serve /path/to/model \
  --backend turbomind \
  --max-concurrent-requests 64 \
  --disable-fastapi-docs \
  --log-level ERROR \
  --session-len 512 \
  --max-batch-size 32 \
  --cache-max-entry-count 0.2 \
  --cache-block-seq-len 64 \
  --max-prefill-token-num 128 \
  --quant-policy 4 \
  --vision-max-batch-size 32
```

---

Bạn có thể gửi thêm:

- Thông tin GPU (A100 / V100 / 4090 / ...),
- Mô hình cụ thể (`internlm-20b`, `llama2-7b`, v.v.),
- Input image resolution

→ Mình sẽ tối ưu lại thông số cho phù hợp nhất.
