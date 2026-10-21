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

## Cài đặt môi trường

Môi trường đã được kiểm tra:

| Thành phần | Phiên bản |
|---|---|
| Python | **3.12** (bắt buộc ≥ 3.11 vì `numpy 2.5`, `pandas 3.0`) |
| torch | 2.13.0 (CUDA 12.6 trên máy dev; xem bên dưới cho GPU đời mới) |
| transformers / accelerate / torchmetrics | 5.15.0 / 1.15.0 / 1.9.0 |
| Các thư viện khác | đã pin trong `requirement.txt` |

### Các bước trên máy GPU (Linux, ví dụ máy thuê)

**1. Kiểm tra driver và CUDA**

```bash
nvidia-smi
```

Xem dòng `CUDA Version` ở góc trên bên phải. Dòng này cho biết phiên bản CUDA cao nhất mà driver hỗ trợ:

Chọn bản build torch theo số này:

| `CUDA Version` trong `nvidia-smi` | Bản torch cần cài | GPU dùng được |
|---|---|---|
| ≥ 13.0 | `cu130` | Mọi GPU, kể cả RTX 50xx |
| 12.9 | `cu129` | Mọi GPU, kể cả RTX 50xx |
| 12.6 – 12.8 | `cu126` | Chỉ RTX 30xx/40xx (ví dụ 3090 Ti) |

**RTX 50xx** (ví dụ 5060 Ti, kiến trúc Blackwell `sm_120`) bắt buộc dùng `cu129` hoặc mới hơn. Nếu máy thuê có 5060 Ti mà `nvidia-smi` chỉ hiện CUDA ≤ 12.8 thì driver quá cũ, nên đổi sang máy khác.

**2. Tạo môi trường Python 3.12**

Image của máy thuê thường có sẵn Python 3.10. Python 3.10 **không** cài được `requirement.txt`, nên cần tạo môi trường riêng.

```bash
# conda (miniconda thường có sẵn trên image PyTorch)
conda create -y -n naml python=3.12
conda activate naml
```

Hoặc dùng `uv`, nếu máy không có conda:

```bash
pip install uv
uv venv --python 3.12 .venv
source .venv/bin/activate
```

Nếu dùng `uv`, thay mọi lệnh `pip install` ở các bước sau bằng `uv pip install`.

**3. Cài torch đúng bản CUDA, sau đó mới cài phần còn lại**

Luôn cài torch trước. Nếu để `pip install -r requirement.txt` tự cài torch, bạn có thể nhận bản CUDA không khớp với GPU.

```bash
# Chọn MỘT dòng theo bảng ở bước 1 (torch 2.13.0 không có bản cu128)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126

pip install -r requirement.txt
```

Sau khi chạy `pip install -r requirement.txt`, kiểm tra torch không bị cài đè. `pip show torch` phải vẫn hiện bản `2.13.0+cuXXX` vừa cài.

**4. Kiểm tra torch thấy GPU**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); torch.ones(1, device='cuda')"
```

Lệnh phải in `True` và tên GPU, không báo lỗi.

**5. Chép dữ liệu đã tiền xử lý lên máy**

Thư mục `data/` không có trên git. Chép bản đã tiền xử lý (khoảng 200MB) từ máy local, giữ đúng đường dẫn:

```bash
# chạy trên máy local, từ thư mục gốc của repo
scp -P <port> -r data/processed/real_data_v1 <user>@<host>:<đường-dẫn-repo>/data/processed/
```

Cách này không cần cài `pyvi` và không phải chạy lại preprocessing trên máy thuê.

**6. Tải trước PhoBERT** (khoảng 540MB, cần internet)

```bash
python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('vinai/phobert-base-v2'); AutoModel.from_pretrained('vinai/phobert-base-v2')"
```

**7. Chạy smoke test trên GPU trước khi train thật**

```bash
python -m scripts.train --config configs/train_smoke.yaml
```

Smoke test chạy khoảng 1 phút. Nếu nó in loss và metric val thì môi trường đã sẵn sàng.

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `Could not find a version that satisfies the requirement numpy==2.5.1` (hoặc `pandas==3.0.5`) | Python < 3.11 | Tạo môi trường Python 3.12 (bước 2) |
| `no kernel image is available for execution on the device` / `sm_120 is not compatible with the current PyTorch installation` | torch build CUDA cũ trên RTX 50xx | `pip uninstall -y torch` rồi cài lại bản `cu129`/`cu130` (bước 3) |
| `torch.cuda.is_available()` trả về `False` | Cài nhầm torch bản CPU, hoặc bản build cần CUDA mới hơn driver hỗ trợ | Cài lại torch theo bảng ở bước 1 và bước 3 |
| `No matching distribution found for torch==2.13.0` khi dùng `--index-url .../cu128` | Không có torch 2.13.0 bản cu128 | Dùng `cu129`, `cu130` hoặc `cu126` |
| `processed_dir ... Path does not point to a directory` | Chưa chép dữ liệu | Bước 5 |
| `... was tokenized with X, but model.pretrained_name is Y` | Dữ liệu processed được tạo bằng tokenizer khác | Dùng đúng bản `real_data_v1`, hoặc chạy lại preprocessing |
| `Bus error` / `DataLoader worker ... killed` / `unable to write to file </torch_...>` | Docker cấp `/dev/shm` quá nhỏ | Đặt `data.num_workers: 0` trong config, hoặc tạo container với `--shm-size=8g` |
| `CUDA out of memory` | Batch quá lớn so với VRAM | Giảm `data.train_batch_size`, tăng `trainer.gradient_accumulation_steps` để giữ batch hiệu dụng; hoặc giảm `model.num_trainable_layers` / `data.history_len` |
| Không tải được `vinai/phobert-base-v2` | Máy không ra được internet tới HuggingFace | Chép thư mục `~/.cache/huggingface/hub/models--vinai--phobert-base-v2` từ máy local sang cùng vị trí trên máy thuê |

### Cài trên Windows (máy dev)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirement.txt
```

Trên Windows, nên để `data.num_workers: 0` nếu gặp lỗi hoặc thấy DataLoader khởi động chậm.

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
