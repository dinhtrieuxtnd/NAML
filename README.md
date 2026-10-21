# NAML cho gợi ý tin tức tiếng Việt

Cài đặt mô hình **NAML** (Neural News Recommendation with Attentive Multi-View Learning, Wu và cộng sự, IJCAI 2019) cho dữ liệu tin tức tiếng Việt. Có hai biến thể, chọn qua config:

| `model.name` | News encoder cho view văn bản | Ghi chú |
|---|---|---|
| `naml` | PhoBERT (`vinai/phobert-base-v2`) + word attention | Đóng băng / train k layer cuối / finetune toàn bộ qua `num_trainable_layers` |
| `naml_cnn` | Embedding PhoBERT + CNN + word attention | Baseline giống bài báo gốc, thay GloVe bằng ma trận embedding của PhoBERT |

Mỗi bài báo có 4 view: `title`, `sapo`, `tags` (văn bản) và `category`. Các view được gộp bằng view-level attention; user vector là news-level attention trên lịch sử click; điểm click là tích vô hướng. Huấn luyện bằng 1 tin dương + K tin âm lấy trong cùng impression.

## Cấu trúc

```
configs/            # mọi hyperparameter (YAML, validate bằng Pydantic)
scripts/            # entry-point: preprocess.py, train.py, evaluate.py
src/data/           # đọc CSV, tiền xử lý, Dataset, collator, hằng số
src/models/         # nn.Module thuần + registry.py (registry duy nhất)
src/engine/         # vòng train (accelerate) và evaluate (torchmetrics)
src/utils/          # tiện ích dùng chung: config loader, I/O
```

## Cài đặt

Python 3.12.

```bash
pip install -r requirement.txt
```

`torch` trong `requirement.txt` là bản trên PyPI; nếu cần bản CUDA cụ thể, cài torch theo hướng dẫn tại pytorch.org trước rồi mới cài phần còn lại.

## Dữ liệu đầu vào

Đặt trong `data/raw/` (thư mục `data/` không được đưa lên git):

- `data/raw/behaviors/{train,val,test}_behaviors.csv`, **không có header**, 5 cột:
  `impression_id, user_id, time, history, impressions`
  - `history`: các `newsId` đã click, cách nhau bởi dấu cách (có thể rỗng).
  - `impressions`: các `newsId-label` cách nhau bởi dấu cách, `label` là 1 (click) hoặc 0.
- `data/raw/news/news.csv`, có header: `newsId, title, sapo, tags, category`.

Ba tập train/val/test phải được chia sẵn theo thời gian, và lịch sử của mỗi impression chỉ gồm các click trước thời điểm impression. Pipeline không kiểm tra lại các điều kiện này.

## 1. Tiền xử lý

```bash
python -m scripts.preprocess --config configs/preprocess.yaml
```

Bước này tách từ bằng ViTokenizer, tokenize bằng tokenizer PhoBERT, rồi ghi vào `data/processed/<name>/`:

- `artifacts/`: token matrix cho từng view (`title.npy`, ...), mẫu `{train,val,test}_samples.pkl`, mapping `newsId → row` và category.
- `config.yaml`, `statistics.json`, `run_info.json`, `preprocess.log`.

Nếu thư mục output đã tồn tại thì script báo lỗi. Đặt `output.overwrite: true` để ghi đè.

## 2. Train

```bash
python -m scripts.train --config configs/train.yaml            # naml (PhoBERT)
python -m scripts.train --config configs/train_naml_cnn.yaml   # baseline naml_cnn
```

Mỗi lần chạy tạo một thư mục `outputs/<run_name>_<timestamp>/`:

```
config.yaml                 # config đã resolve của run
tensorboard/                # loss train, metric val
checkpoints/epoch_<i>/      # sau mỗi epoch (giữ lại tất cả)
checkpoints/latest/         # ghi đè mỗi trainer.save_every_steps step, dùng để resume
```

Sau mỗi epoch, script đánh giá trên tập val. Tập test không được dùng trong lúc train.

Theo dõi quá trình train:

```bash
tensorboard --logdir outputs
```

### Resume

Nếu máy bị ngắt, chạy lại với `--resume`, trỏ tới checkpoint muốn train tiếp:

```bash
python -m scripts.train --config configs/train.yaml --resume outputs/<run>/checkpoints/latest
```

Run được resume tiếp tục ghi vào thư mục run cũ. Thứ tự dữ liệu, tin âm và RNG đều cố định theo seed, nên kết quả giống hệt như khi chạy liên tục. Được phép đổi các tham số như `epochs`, `max_train_steps`, `num_workers`, `eval_batch_size`. Nếu đổi `model`, `loss`, `optimizer`, dữ liệu train, `seed` hoặc `gradient_accumulation_steps` thì script báo lỗi.

## 3. Đánh giá checkpoint

Chọn checkpoint tốt nhất theo metric val trên tensorboard, rồi chạy trên tập test:

```bash
python -m scripts.evaluate --checkpoint outputs/<run>/checkpoints/epoch_<i> --split test
```

Kết quả được ghi vào `<checkpoint>/metrics_test.json`. Các metric (tính bằng torchmetrics):

| Metric | Ý nghĩa |
|---|---|
| `auc` | AUC trên toàn bộ cặp (user, candidate) |
| `impression_auc` | AUC trung bình theo từng impression (cách bài báo gốc báo cáo) |
| `mrr` | Mean Reciprocal Rank |
| `ndcg@5`, `ndcg@10` | nDCG tại top 5 / 10 |

Các impression có lịch sử rỗng bị loại khỏi train và evaluate.

## Config chính

| Trường | Ý nghĩa |
|---|---|
| `data.history_len` | Số tin gần nhất trong lịch sử được dùng |
| `data.num_negatives` | K tin âm mỗi mẫu dương (thiếu thì lấy lặp lại) |
| `data.max_eval_impressions` | Giới hạn số impression val/test (`null` = toàn bộ) |
| `model.num_trainable_layers` | `naml`: 0 = đóng băng PhoBERT, k = train k layer cuối, 12 = finetune toàn bộ |
| `model.freeze_embedding` | `naml_cnn`: đóng băng ma trận embedding hay không |
| `optimizer.lr` / `optimizer.encoder_lr` | Learning rate cho phần head / phần pretrained |
| `trainer.mixed_precision` | `no`, `fp16` hoặc `bf16` |
| `trainer.max_train_steps` | Dừng sau N optimizer step (`null` = chạy đủ `epochs`) |
| `trainer.save_every_steps` | Chu kỳ ghi `checkpoints/latest` (`null` = chỉ lưu cuối epoch) |

Thêm model hoặc loss mới: viết module trong `src/models/`, thêm config Pydantic, rồi đăng ký vào `src/models/registry.py`. Không nhân bản `train.py`.

## Smoke test

Chạy sau mỗi thay đổi trong `src/` hoặc `configs/`. Lệnh dưới train 5 step và đánh giá trên 20 impression, không cần GPU mạnh:

```bash
python -m scripts.train --config configs/train_smoke.yaml
```

## Tham khảo

C. Wu, F. Wu, M. An, J. Huang, Y. Huang, X. Xie. *Neural News Recommendation with Attentive Multi-View Learning*. IJCAI 2019. https://arxiv.org/abs/1907.05576
