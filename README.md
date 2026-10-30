# NAML cho gợi ý tin tức tiếng Việt

Cài đặt mô hình **NAML** (Neural News Recommendation with Attentive Multi-View Learning, Wu và cộng sự, IJCAI 2019) cho dữ liệu tin tức tiếng Việt, theo đúng kiến trúc của bài báo gốc. Word embedding là fastText `cc.vi.300` (300 chiều) thay cho GloVe. Model có khoảng 9 triệu tham số, trong đó khoảng 7,4 triệu thuộc ma trận embedding.

Mỗi bài báo có 4 view: `title`, `sapo`, `tags` (văn bản) và `category`.
- Mỗi view văn bản đi qua word embedding, CNN rồi word attention.
- `category` đi qua embedding rồi một lớp dense.
- Các view được gộp bằng view-level attention.
- User vector là news-level attention trên lịch sử click; điểm click là tích vô hướng.
- Huấn luyện bằng 1 tin dương + K tin âm lấy trong cùng impression.

Config mặc định dùng biến thể **`naml_popularity`**: cộng thêm vào điểm NAML một nhánh popularity theo thời gian thực.
- `c_i`: số click vào candidate `i` trong cửa sổ 15 phút ngay trước impression.
- `c̃_i = log(1 + c_i) − mean_j log(1 + c_j)`: độ hot tương đối so với các candidate khác trong cùng impression.
- `t = [sin(2πh/24), cos(2πh/24)]`, với `h` là giờ của impression (giờ + phút/60).
- `q_i = MLP([t, Linear(u), c̃_i])`: thời điểm, user và độ hot được kết hợp đồng thời, nên mỗi user và mỗi thời điểm có một cách dùng popularity riêng.
- Điểm cuối: `s_i = u·n_i + q_i`.

Đặt `model.name: naml` (và bỏ hai trường `popularity_*`) để train NAML gốc.

## Cấu trúc

```
configs/            # mọi hyperparameter (YAML, validate bằng Pydantic)
scripts/            # entry-point: preprocess.py, train.py, evaluate.py, baselines.py
src/data/           # đọc CSV, tách token, embedding, tiền xử lý, Dataset, collator, hằng số
src/models/         # nn.Module thuần + registry.py (registry duy nhất)
src/engine/         # vòng train (accelerate), evaluate + metric (torchmetrics), baseline, vẽ biểu đồ
src/utils/          # tiện ích dùng chung: config loader, I/O, quy chuẩn run (run.py)
```

## Cài đặt và chạy

Môi trường đã được kiểm tra:

| Thành phần | Phiên bản |
|---|---|
| Python | **3.12** (bắt buộc ≥ 3.11 vì `numpy 2.5`, `pandas 3.0`) |
| torch | 2.13.0 (cài riêng, không có trong `requirement.txt`) |
| accelerate / torchmetrics / matplotlib | 1.15.0 / 1.9.0 / 3.11.1 |
| Các thư viện khác | đã pin trong `requirement.txt` |

Dữ liệu tải về là bản **đã tiền xử lý** (`data/processed/real_data_fasttext_v3`, đã gồm `embedding.npy` và popularity của từng candidate). Vì vậy không cần file fastText 4,5GB và không phải chạy lại bước [tiền xử lý](#1-tiền-xử-lý) trước khi train.

### Linux (ví dụ máy GPU thuê)

**1. Clone repo**

```bash
git clone https://github.com/dinhtrieuxtnd/NAML.git
cd NAML
```

**2. Tạo môi trường Python 3.12**

Image của máy thuê thường chỉ có Python 3.10, mà Python 3.10 **không** cài được `requirement.txt`.

```bash
sudo apt install python3.12 python3.12-venv
python3.12 -m venv venv
source venv/bin/activate
```

Các bước sau đều chạy trong môi trường này (dòng lệnh có tiền tố `(venv)`).

**3. Cài torch, sau đó mới cài phần còn lại**

Xem dòng `CUDA Version` ở góc trên bên phải của `nvidia-smi` để chọn bản torch:

| `CUDA Version` trong `nvidia-smi` | Bản torch | GPU dùng được |
|---|---|---|
| ≥ 13.0 | `cu130` | Mọi GPU, kể cả RTX 50xx |
| 12.9 | `cu129` | Mọi GPU, kể cả RTX 50xx |
| 12.6 – 12.8 | `cu126` | Chỉ RTX 30xx/40xx |

```bash
# đổi cu126 theo bảng trên (torch 2.13.0 không có bản cu128)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirement.txt
```

Kiểm tra torch thấy GPU. Lệnh phải in `True` và tên GPU, không báo lỗi:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**4. Tải và giải nén dữ liệu**

```bash
pip install gdown
gdown "https://drive.google.com/file/d/1xldSpS63doRgA94rRoSjFgY-Iwy2CMpJ/view?usp=sharing"
sudo apt install unrar
unrar x data.rar data/processed
```

Sau khi giải nén phải có thư mục `data/processed/real_data_fasttext_v3/`, đúng với `data.processed_dir` trong `configs/train.yaml`.

**5. Smoke test, train, đánh giá**

```bash
# kiểm tra môi trường, chạy dưới 1 phút
python -m scripts.train --config configs/train_smoke.yaml
# train thật
python -m scripts.train --config configs/train.yaml
# đánh giá checkpoint tốt nhất trên tập test
python -m scripts.evaluate --checkpoint outputs/<run>/checkpoints/best
```

Smoke test in vài dòng `epoch=...` và `train completed` là môi trường đã sẵn sàng. Chi tiết từng bước xem [Train](#2-train) và [Đánh giá](#3-đánh-giá-trên-tập-test).

### Windows

1. **Cài Python 3.12:** tải bộ cài từ [python.org/downloads](https://www.python.org/downloads/). Ở màn hình đầu tiên của bộ cài, **tick ô "Add python.exe to PATH"** rồi bấm *Install Now*.
2. **Tải code:** trên trang GitHub của repo, bấm nút **Code → Download ZIP**, rồi giải nén ra một thư mục, ví dụ `D:\NAML`.
3. **Tải dữ liệu:** mở [link Google Drive](https://drive.google.com/file/d/1xldSpS63doRgA94rRoSjFgY-Iwy2CMpJ/view?usp=sharing) bằng trình duyệt và bấm **Download**. Giải nén `data.rar` bằng WinRAR hoặc 7-Zip (chuột phải → *Extract files…*) vào thư mục `data\processed` bên trong thư mục code. Kết quả phải có `D:\NAML\data\processed\real_data_fasttext_v3\`.
4. **Mở terminal:** mở thư mục code bằng VS Code (*File → Open Folder…*), rồi chọn *Terminal → New Terminal*.
5. **Cài thư viện:** chạy lần lượt trong terminal:
   ```bash
   py -3.12 -m venv venv
   venv\Scripts\activate
   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirement.txt
   ```
   Nếu VS Code hỏi *"We noticed a new environment has been created…"* thì bấm **Yes** để dùng môi trường `venv` này.
6. **Chạy:** dùng đúng các lệnh ở bước 5 của phần Linux.

Trên Windows nên đặt `data.num_workers: 0` trong config nếu gặp lỗi hoặc thấy DataLoader khởi động chậm.

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `Could not find a version that satisfies the requirement numpy==2.5.1` (hoặc `pandas==3.0.5`) | Python < 3.11 | Tạo môi trường Python 3.12 (bước 2) |
| `no kernel image is available for execution on the device` / `sm_120 is not compatible with the current PyTorch installation` | torch build CUDA cũ trên RTX 50xx | `pip uninstall -y torch` rồi cài lại bản `cu129`/`cu130` (bước 3) |
| `torch.cuda.is_available()` trả về `False` | Cài nhầm torch bản CPU, hoặc bản build cần CUDA mới hơn driver hỗ trợ | Cài lại torch theo bảng ở bước 3 |
| `No matching distribution found for torch==2.13.0` khi dùng `--index-url .../cu128` | Không có torch 2.13.0 bản cu128 | Dùng `cu129`, `cu130` hoặc `cu126` |
| `processed_dir ... Path does not point to a directory` | Chưa giải nén dữ liệu, hoặc giải nén sai chỗ | Bước 4: phải có `data/processed/real_data_fasttext_v3/` |
| `KeyError: 'popularity'` (hoặc `'hour'`) | `processed_dir` trỏ tới bản dữ liệu cũ (v1, v2) không có popularity | Dùng `real_data_fasttext_v3`, hoặc chạy lại tiền xử lý |
| `Bus error` / `DataLoader worker ... killed` / `unable to write to file </torch_...>` | Docker cấp `/dev/shm` quá nhỏ | Đặt `data.num_workers: 0` trong config, hoặc tạo container với `--shm-size=8g` |
| `CUDA out of memory` | Batch quá lớn so với VRAM | Giảm `data.train_batch_size`, tăng `trainer.gradient_accumulation_steps` để giữ batch hiệu dụng; hoặc giảm `data.history_len` |
| `Cannot resume ...: config differs in ...` | Đổi trường cố định khi resume | Giữ nguyên các trường được liệt kê (xem [Resume](#resume)) |

## Dữ liệu đầu vào

Đặt trong `data/raw/` (thư mục `data/` không được đưa lên git):

- `data/raw/behaviors/{train,val,test}_behaviors.csv`, **không có header**, 5 cột:
  `impression_id, user_id, time, history, impressions`
  - `history`: các `newsId` đã click, cách nhau bởi dấu cách (có thể rỗng).
  - `impressions`: các `newsId-label` cách nhau bởi dấu cách, `label` là 1 (click) hoặc 0.
- `data/raw/news/news.csv`, có header: `newsId, title, sapo, tags, category`.
- `data/raw/fasttext/cc.vi.300.vec`: word vector fastText tiếng Việt (4,5GB, định dạng `.vec` có dòng header). Chỉ cần lúc tiền xử lý. Trên Windows có thể tạo link thay vì chép:
  ```bash
  mklink data\raw\fasttext\cc.vi.300.vec <đường-dẫn-tới>\cc.vi.300.vec
  ```
- `create_data/raw_log_cleaned/`: log sự kiện đã làm sạch (parquet), dùng để đếm click cho popularity (eventId 2, theo giờ click thật `clientTime`). Chỉ đọc, không sửa.

Ba tập train/val/test phải được chia sẵn theo thời gian, và lịch sử của mỗi impression chỉ gồm các click trước thời điểm impression. Pipeline không kiểm tra lại các điều kiện này.

## Quy chuẩn run

Mỗi lần chạy một script là một run, và config nào cũng có section `run: {name, output_root}`.

| Stage | Thư mục | Các file chung |
|---|---|---|
| Tiền xử lý | `output_root/name`, tên cố định vì config train trỏ tới nó | `config.yaml`, `run_info.json`, `preprocess.log` |
| Train | `output_root/<name>_<timestamp>`; resume ghi tiếp vào thư mục cũ | `config.yaml`, `run_info.json`, `train.log` |
| Evaluate | `<thư mục run train>/evaluation_<checkpoint>/` | chỉ `metrics.json` và `predictions.csv`; log chỉ in ra màn hình |
| Baseline | `output_root/<name>_<timestamp>` | `config.yaml`, `run_info.json`, `baselines.log`, `metrics.json` |

- `config.yaml` là config đã resolve và không bao giờ bị ghi đè.
- `run_info.json` là danh sách, mỗi lần chạy hoặc resume thêm một mục, gồm:
  - stage, trạng thái (`running`/`completed`/`failed` kèm lỗi), thời gian;
  - phiên bản python/torch/CUDA, GPU, git commit, câu lệnh.
- Log ghi ra cả file và màn hình nhưng chỉ ghi các mốc quan trọng. Tiến độ theo step chỉ hiện bằng thanh `tqdm` trên terminal, không ghi vào file log.

## 1. Tiền xử lý

```bash
python -m scripts.preprocess --config configs/preprocess.yaml
```

Toàn bộ bước này mất khoảng 1–2 phút.
- **Tách token:** NFKC → chữ thường → regex `\w+|[^\w\s]`. Tách theo âm tiết, mỗi dấu câu là một token. Không dùng tách từ ghép, vì fastText `cc.vi` không có vector cho từ ghép.
- **Từ điển:** dựng từ text của news. `<PAD>` = 0, `<UNK>` = 1, các token còn lại xếp theo tần suất giảm dần.
- **Ma trận embedding:** dựng một lần từ file `.vec`, đọc theo từng dòng. Mỗi token được tra như sau:
  - Ưu tiên khớp chính xác.
  - Sau đó thử dạng viết hoa.
  - Nếu vẫn không có thì khởi tạo theo `N(0, embedding.unmatched_init_std)`.
  - Hàng `<PAD>` bằng 0.
- **Lịch sử click:** bỏ tin trùng, giữ lần click gần nhất.
- **Popularity:** với mỗi impression, đếm click vào từng candidate trong `popularity.window_minutes` phút ngay trước thời điểm impression, chỉ lấy click xảy ra trước đó nên không rò rỉ tương lai. Sau đó tính `c̃` trên **toàn bộ** candidate của impression, trước khi lấy mẫu tin âm, để train và evaluate dùng cùng một giá trị. Giờ của impression cũng được lưu vào mẫu.

Kết quả ghi vào `data/processed/<run.name>/`:

- `artifacts/`:
  - token matrix cho từng view (`title.npy`, ...);
  - `vocab.json`, `embedding.npy`;
  - mẫu `{train,val,test}_samples.pkl`, kèm popularity của từng candidate và giờ của impression;
  - mapping `newsId → row` và category.
- `statistics.json`:
  - kích thước vocab;
  - độ phủ embedding theo loại từ (~89%) và theo token (~99,7%);
  - phân bố độ dài theo âm tiết (`p95`, tỉ lệ bị cắt), dùng để chỉnh `text.max_length`;
  - số mẫu và số click trùng đã bỏ;
  - số candidate và số candidate không có click nào trong cửa sổ (khoảng 32%).
- `config.yaml`, `run_info.json`, `preprocess.log`.

Nếu thư mục output đã tồn tại thì script báo lỗi. Đặt `run.overwrite: true` để ghi đè.

## 2. Train

```bash
python -m scripts.train --config configs/train.yaml
```

- Optimizer là AdamW, có gradient clipping (`trainer.max_grad_norm`).
- Learning rate giảm theo `ReduceLROnPlateau` trên metric val `trainer.monitor` (mặc định `ndcg@10`); scheduler step một lần mỗi epoch.
- Early stopping dùng cùng metric đó. Train dừng khi `early_stopping.patience` epoch liên tiếp không vượt giá trị tốt nhất cộng `min_delta`.

Sau mỗi epoch, script đánh giá trên tập val. Tập test không được dùng trong lúc train. Mỗi epoch ghi đúng một dòng log:

```
epoch=2 loss=1.4472 auc=0.4931 imp_auc=0.6084 mrr=0.4829 ndcg@5=0.5410 ndcg@10=0.5817 lr=1.00e-04 time=0.9s best
```

Thư mục run `outputs/<run.name>_<timestamp>/`:

```
config.yaml, run_info.json, train.log
history.json                # mỗi epoch một dòng: epoch, train_loss, lr, time, các metric val
plots/                      # loss.png, val_<metric>.png mỗi metric một hình (đánh dấu epoch tốt nhất), lr.png; vẽ lại sau mỗi epoch
checkpoints/best/           # tốt nhất theo trainer.monitor, kèm metrics.json
checkpoints/latest/         # ghi đè mỗi trainer.save_every_steps step và cuối mỗi epoch, dùng để resume
evaluation_<checkpoint>/    # tạo bởi scripts/evaluate.py
```

Mỗi checkpoint chứa model, optimizer, scheduler, RNG, `config.yaml` của nó và `training_state.json`. File `training_state.json` lưu:
- vị trí đang train;
- loss đang cộng dồn của epoch dở;
- giá trị tốt nhất và bộ đếm patience;
- `history`.

### Resume

Nếu máy bị ngắt, chạy lại với `--resume`, trỏ tới checkpoint muốn train tiếp:

```bash
python -m scripts.train --config configs/train.yaml --resume outputs/<run>/checkpoints/latest
```

Run được resume ghi tiếp vào thư mục run cũ.
- **Kết quả giống hệt chạy liên tục:** thứ tự dữ liệu, tin âm và RNG cố định theo (seed, epoch), và evaluate không làm lệch RNG. Điều này đúng cả khi dừng giữa epoch lẫn ở ranh giới epoch.
- **Được đổi:** `epochs`, `max_train_steps`, `save_every_steps`, `num_workers`, `eval_batch_size`, `early_stopping`.
- **Đổi thì script báo lỗi:**
  - `model`, `loss`, `optimizer`, `scheduler`;
  - dữ liệu train: `processed_dir`, `text_views`, `history_len`, `num_negatives`, `train_batch_size`, `max_train_samples`;
  - `seed`, `gradient_accumulation_steps`, `monitor`.

## 3. Đánh giá trên tập test

Sau khi train xong, đánh giá checkpoint đã chọn (thường là `best`) trên tập test:

```bash
python -m scripts.evaluate --checkpoint outputs/<run>/checkpoints/best
```

Config được đọc từ `config.yaml` của checkpoint. Kết quả ghi vào `outputs/<run>/evaluation_best/`:

- `metrics.json`: các metric bên dưới, tính bằng torchmetrics.
- `predictions.csv`: mỗi dòng là một cặp impression–candidate, gồm `impression_id, user_id, news_id, label, score, rank`. `rank` = 1 là điểm cao nhất trong impression.

| Metric | Ý nghĩa |
|---|---|
| `auc` | AUC trên toàn bộ cặp (user, candidate) |
| `impression_auc` | AUC trung bình theo từng impression (cách bài báo gốc báo cáo) |
| `mrr` | Mean Reciprocal Rank |
| `ndcg@5`, `ndcg@10` | nDCG tại top 5 / 10 |

Các impression có lịch sử rỗng bị loại khỏi train và evaluate.

## 4. Baseline

Để biết model học được bao nhiêu, so sánh với hai cách xếp hạng không học, chấm trên đúng các impression val/test mà model được đánh giá:

```bash
python -m scripts.baselines --config configs/baselines.yaml
```

- `random`: điểm ngẫu nhiên cho mỗi candidate (seed cố định).
- `popularity`: số click vào candidate trong `popularity_window_minutes` phút ngay trước thời điểm impression. Click được đếm từ log sự kiện đã làm sạch (`click_log`, eventId 2) theo giờ click thật `clientTime`, chỉ lấy click xảy ra trước impression, nên không rò rỉ tương lai. Không đếm từ behaviors CSV vì ở đó mọi click của một session đều mang giờ bắt đầu session. Hoà điểm được phá ngẫu nhiên.

Baseline cần thời gian chính xác của impression, mà dữ liệu processed chỉ lưu giờ trong ngày, nên config trỏ thêm tới các file behaviors val/test thô đã dùng để tạo `processed_dir`. Nên giữ `popularity_window_minutes` bằng `popularity.window_minutes` của `configs/preprocess.yaml`, để baseline và model dùng cùng cửa sổ. Kết quả ghi vào `outputs/<run.name>_<timestamp>/metrics.json` theo dạng `{split: {baseline: {metric: value}}}`. Baseline không train gì nên được chấm cả val và test. `popularity_window_minutes` chỉ nên chọn theo val.

## Config chính

| Trường | Ý nghĩa |
|---|---|
| `text.max_length.{title,sapo,tags}` | Độ dài tối đa (số token âm tiết) của từng view, chỉnh theo `p95` trong `statistics.json` |
| `embedding.unmatched_init_std` | Độ lệch chuẩn khởi tạo cho token không có trong fastText |
| `popularity.window_minutes` | Cửa sổ đếm click cho popularity lúc tiền xử lý (15 phút, tốt nhất trên val); đổi thì phải tiền xử lý lại |
| `model.name` | `naml_popularity` (mặc định) hoặc `naml` (NAML gốc) |
| `model.popularity_user_dim` | Số chiều của user sau khi chiếu, trước khi ghép với thời điểm và popularity |
| `model.popularity_hidden_dim` | Số chiều lớp ẩn của MLP popularity |
| `data.history_len` | Số tin gần nhất trong lịch sử được dùng (sau khi bỏ tin trùng) |
| `data.num_negatives` | K tin âm mỗi mẫu dương (thiếu thì lấy lặp lại) |
| `data.max_train_samples` / `data.max_eval_impressions` | Giới hạn số mẫu train / impression val-test, dùng cho smoke test (`null` = toàn bộ) |
| `model.freeze_embedding` | Đóng băng ma trận word embedding hay không |
| `model.num_filters` / `model.window_size` | Số filter và kích thước cửa sổ của CNN |
| `optimizer.lr` / `optimizer.weight_decay` | AdamW, cùng một lr cho mọi tham số |
| `scheduler.{factor,patience,min_lr}` | `ReduceLROnPlateau` theo `trainer.monitor` |
| `early_stopping.{patience,min_delta}` | Dừng khi metric không cải thiện |
| `trainer.monitor` | Metric val để chọn `best`, cho scheduler và early stopping |
| `trainer.max_grad_norm` | Ngưỡng gradient clipping |
| `trainer.mixed_precision` | `no`, `fp16` hoặc `bf16` |
| `trainer.max_train_steps` | Dừng sau N optimizer step (`null` = chạy đủ `epochs`) |
| `trainer.save_every_steps` | Chu kỳ ghi `checkpoints/latest` (`null` = chỉ lưu cuối epoch) |

Thêm model hoặc loss mới: viết module trong `src/models/`, thêm config Pydantic, rồi đăng ký vào `src/models/registry.py`. Không nhân bản `train.py`.

## Smoke test

Chạy sau mỗi thay đổi trong `src/` hoặc `configs/`. Lệnh dưới train tối đa 3 epoch trên 64 mẫu và đánh giá val trên 20 impression; early stopping (`patience` 1) có thể dừng sớm hơn. Không cần GPU mạnh:

```bash
python -m scripts.train --config configs/train_smoke.yaml
```

## Tham khảo

C. Wu, F. Wu, M. An, J. Huang, Y. Huang, X. Xie. *Neural News Recommendation with Attentive Multi-View Learning*. IJCAI 2019. https://arxiv.org/pdf/1907.05576
