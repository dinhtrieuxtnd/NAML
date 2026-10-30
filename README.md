# NAML cho gợi ý tin tức tiếng Việt

Cài đặt mô hình **NAML** (Wu và cộng sự, IJCAI 2019) cho tin tức tiếng Việt, dùng word embedding fastText `cc.vi.300`.
- Mỗi bài báo có 4 view: `title`, `sapo`, `tags` (CNN + word attention) và `category`, gộp bằng view-level attention thành `n`.
- User `u` là attention trên lịch sử click. Huấn luyện bằng 1 tin dương + K tin âm trong cùng impression.

Config mặc định dùng biến thể **`naml_popularity`**, cộng thêm một nhánh popularity theo thời gian thực:
- `c̃_i = log(1 + c_i) − mean_j log(1 + c_j)`, với `c_i` là số click vào candidate `i` trong 15 phút trước impression.
- `t = [sin(2πh/24), cos(2πh/24)]`, với `h` là giờ của impression.
- `s_i = u·n_i + MLP([t, Linear(u), c̃_i])`.

Đặt `model.name: naml` (và bỏ hai trường `popularity_*`) để train NAML gốc.

```
configs/      # mọi hyperparameter (YAML, validate bằng Pydantic)
scripts/      # entry-point: preprocess, train, evaluate, baselines
src/data/     # đọc dữ liệu, tiền xử lý, popularity, Dataset, collator, hằng số
src/models/   # nn.Module + registry.py
src/engine/   # train (accelerate), evaluate (torchmetrics), baseline, biểu đồ
src/utils/    # config, I/O, khởi tạo run
```

## Cài đặt

Cần **Python 3.12** (≥ 3.11) và torch 2.13.0 (cài riêng, không có trong `requirement.txt`). Dữ liệu tải về đã được tiền xử lý sẵn (`data/processed/real_data_fasttext_v3`), nên có thể train ngay.

### Linux

Chạy lần lượt các lệnh trong [bash.md](bash.md). Bản torch chọn theo dòng `CUDA Version` của `nvidia-smi`:

| `CUDA Version` | Bản torch | GPU |
|---|---|---|
| ≥ 13.0 | `cu130` | Mọi GPU, kể cả RTX 50xx |
| 12.9 | `cu129` | Mọi GPU, kể cả RTX 50xx |
| 12.6 – 12.8 | `cu126` | Chỉ RTX 30xx/40xx |

### Windows

1. **Cài Python 3.12** từ [python.org/downloads](https://www.python.org/downloads/). Ở màn hình đầu của bộ cài, **tick "Add python.exe to PATH"** rồi bấm *Install Now*.
2. **Tải code:** trên trang GitHub của repo, bấm **Code → Download ZIP** rồi giải nén, ví dụ ra `D:\NAML`.
3. **Tải dữ liệu** từ [Google Drive](https://drive.google.com/file/d/1xldSpS63doRgA94rRoSjFgY-Iwy2CMpJ/view?usp=sharing), rồi giải nén `data.rar` bằng WinRAR hoặc 7-Zip (chuột phải → *Extract files…*) vào `data\processed` trong thư mục code. Kết quả phải có `D:\NAML\data\processed\real_data_fasttext_v3\`.
4. **Mở thư mục code bằng VS Code** (*File → Open Folder…*), chọn *Terminal → New Terminal* và chạy:
   ```bash
   py -3.12 -m venv venv
   venv\Scripts\activate
   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirement.txt
   ```
   Nếu VS Code hỏi có dùng môi trường mới không thì bấm **Yes**.
5. **Chạy** smoke test, train và evaluate bằng các lệnh ở phần [Chạy](#chạy). Nếu DataLoader lỗi hoặc khởi động chậm, đặt `data.num_workers: 0` trong config.

### Lỗi thường gặp

| Lỗi | Cách sửa |
|---|---|
| `Could not find a version ... numpy==2.5.1` | Python < 3.11: tạo lại môi trường bằng Python 3.12 |
| `torch.cuda.is_available()` là `False`, hoặc `no kernel image is available` trên RTX 50xx | Cài lại torch đúng bản `cuXXX` theo bảng trên |
| `processed_dir ... Path does not point to a directory` | Chưa giải nén dữ liệu, hoặc giải nén sai chỗ |
| `KeyError: 'popularity'` | `processed_dir` trỏ tới bản dữ liệu cũ; dùng `real_data_fasttext_v3` |
| `CUDA out of memory` | Giảm `data.train_batch_size` và tăng `trainer.gradient_accumulation_steps` |

## Chạy

```bash
# kiểm tra môi trường (dưới 1 phút); chạy lại sau mỗi thay đổi trong src/ hoặc configs/
python -m scripts.train --config configs/train_smoke.yaml
# train
python -m scripts.train --config configs/train.yaml
# train tiếp khi bị ngắt (kết quả giống hệt chạy liên tục)
python -m scripts.train --config configs/train.yaml --resume outputs/<run>/checkpoints/latest
# đánh giá trên tập test
python -m scripts.evaluate --checkpoint outputs/<run>/checkpoints/best
```

**Train** đánh giá trên val sau mỗi epoch, in một dòng `epoch=... loss=... ndcg@10=... lr=...` và dừng sớm theo `trainer.monitor` (mặc định `ndcg@10`). Tập test không được dùng lúc train. Thư mục run `outputs/<run.name>_<timestamp>/` gồm:
- `train.log`, `history.json` (metric từng epoch) và `plots/`;
- `checkpoints/best/` (kèm `metrics.json`) và `checkpoints/latest/` (để resume).

**Evaluate** đọc config từ checkpoint và ghi vào `outputs/<run>/evaluation_best/`:
- `metrics.json`: `auc`, `impression_auc`, `mrr`, `ndcg@5`, `ndcg@10`;
- `predictions.csv`: mỗi dòng một cặp impression–candidate kèm `score` và `rank`.

### Tiền xử lý lại từ dữ liệu thô

Chỉ cần khi đổi dữ liệu, `text.max_length` hoặc `popularity.window_minutes`. Các file đầu vào:
- `data/raw/behaviors/{train,val,test}_behaviors.csv`, đã chia theo thời gian;
- `data/raw/news/news.csv`;
- `data/raw/fasttext/cc.vi.300.vec` (4,5GB);
- `create_data/raw_log_cleaned/` (log click để đếm popularity).

```bash
python -m scripts.preprocess --config configs/preprocess.yaml
```

Kết quả ghi vào `data/processed/<run.name>/`. Thống kê như độ phủ embedding, độ dài văn bản và tỉ lệ candidate không có click nằm trong `statistics.json`. Script báo lỗi nếu thư mục đã tồn tại, trừ khi đặt `run.overwrite: true`.

### Baseline

```bash
python -m scripts.baselines --config configs/baselines.yaml
```

Lệnh này chấm hai cách xếp hạng không học trên val/test: `random` và `popularity` (số click trong `popularity_window_minutes` phút trước impression). Nên giữ cửa sổ này bằng `popularity.window_minutes` của `configs/preprocess.yaml`.

## Config chính

| Trường | Ý nghĩa |
|---|---|
| `model.name` | `naml_popularity` (mặc định) hoặc `naml` |
| `model.popularity_user_dim` / `popularity_hidden_dim` | Số chiều user sau khi chiếu / lớp ẩn của MLP popularity |
| `model.freeze_embedding` | Đóng băng word embedding hay không |
| `data.history_len` / `data.num_negatives` | Số tin lịch sử gần nhất / số tin âm mỗi mẫu dương |
| `data.max_train_samples` / `max_eval_impressions` | Giới hạn số mẫu, dùng cho smoke test (`null` = toàn bộ) |
| `optimizer.lr` / `weight_decay` | AdamW |
| `early_stopping.{patience,min_delta}` | Dừng khi `trainer.monitor` không cải thiện |
| `trainer.mixed_precision` | `no`, `fp16` hoặc `bf16` |
| `popularity.window_minutes` (preprocess) | Cửa sổ đếm click; đổi thì phải tiền xử lý lại |

Thêm model hoặc loss mới: viết module trong `src/models/`, thêm config Pydantic và đăng ký vào `src/models/registry.py`.

## Tham khảo

C. Wu, F. Wu, M. An, J. Huang, Y. Huang, X. Xie. *Neural News Recommendation with Attentive Multi-View Learning*. IJCAI 2019. https://arxiv.org/pdf/1907.05576
