# Kế hoạch refactor NAML theo CLAUDE.md mới

> Sau khi duyệt, nội dung này được lưu vào project tại `docs/refactor_plan.md`. Code sẽ được sửa **theo từng giai đoạn, khi bạn yêu cầu**.

## Context
CLAUDE.md vừa được cập nhật với các quy tắc mới, nhưng code hiện tại (commit `8e966f1`) vẫn làm theo cách cũ:
- Dùng PhoBERT, `pyvi` và `transformers`; trong khi §7 yêu cầu fastText `cc.vi.300` và tách theo âm tiết. Phép đo trên dữ liệu thật: tách theo âm tiết phủ 99,7% token, ViTokenizer chỉ phủ 75,7%.
- Dùng tensorboard và log theo step; trong khi §11 yêu cầu `history.json`, biểu đồ matplotlib, mỗi epoch một dòng log và tqdm.
- Chưa có early stopping, scheduler hay gradient clipping (§8). Checkpoint vẫn theo kiểu `epoch_<i>`.
- Preprocess và train mỗi bên lưu output một kiểu, chưa có quy chuẩn run chung (§10).
- Evaluate ghi kết quả vào trong thư mục checkpoint và chưa xuất predictions (§10).

**Kết quả mong muốn:** một model NAML dùng fastText, nhẹ (khoảng 10 triệu tham số) và train nhanh. Ba stage preprocess, train và evaluate dùng chung một quy chuẩn run. Log gọn, có biểu đồ, có early stopping, và resume vẫn khớp chính xác như hiện nay.

## Quyết định đã chốt
- Xoá hẳn bản PhoBERT. Model duy nhất là `naml` (CNN cùng embedding fastText, đúng như bài báo gốc).
- `monitor` mặc định là `ndcg@10`, dùng chung cho early stopping và scheduler.
- Luôn bỏ tin trùng trong lịch sử (giữ lần click gần nhất). Không có cờ config để tắt.
- Biểu đồ lưu ở `plots/` của thư mục run. Evaluate chỉ chạy tập test, xuất `metrics.json` và `predictions.csv`.

---

## Giai đoạn 1: Nền tảng chung (quy chuẩn run, logging, thư viện)
Mục tiêu: tạo hạ tầng mà mọi stage dùng chung. Model và dữ liệu chưa đổi, smoke test với PhoBERT vẫn phải chạy được.

1. **`requirement.txt`**: gỡ `transformers`, `pyvi`, `tensorboard`; thêm `matplotlib==3.11.1` (bản đang cài trên máy dev). Việc gỡ `transformers` và `pyvi` làm ở cuối giai đoạn 2, khi code không còn import chúng.
2. **`src/utils/run.py`** (mới):
   - Hằng số layout của run, định nghĩa tại một chỗ duy nhất: `CONFIG_FILE`, `RUN_INFO_FILE`, `HISTORY_FILE`, `PLOTS_DIR`, `CHECKPOINTS_DIR`, `BEST_CHECKPOINT`, `LATEST_CHECKPOINT`, `TRAINING_STATE_FILE`, `METRICS_FILE`, `PREDICTIONS_FILE`, `EVALUATION_DIR_TEMPLATE = "evaluation_{checkpoint}"`, `LOG_FILE_TEMPLATE = "{stage}.log"`.
   - Chuyển các hằng số layout run đang đặt trong `src/data/constants.py` sang đây. Trong `constants.py` chỉ giữ các hằng số về dữ liệu và artifact.
   - `setup_logging(log_file: Path | None)`: gọi `logging.basicConfig` đúng một lần, ghi ra màn hình và file nếu có `log_file`. Đây là chỗ duy nhất trong project cấu hình logging. Thư viện bên ngoài log từ mức WARNING, logger `src.*` log từ mức INFO.
   - `start_run(stage, run_dir, config, resume)` (context manager):
     - Tạo `run_dir`.
     - Ghi `config.yaml` nếu là run mới (không bao giờ ghi đè).
     - Gọi `setup_logging(run_dir / "<stage>.log")`.
     - Thêm một mục vào `run_info.json`: trạng thái, thời điểm bắt đầu/kết thúc, thời lượng, câu lệnh, phiên bản python/torch/CUDA, GPU, git commit (`git rev-parse HEAD`, ghi `null` nếu máy không có git).
     - Khi có exception: ghi `status: failed` kèm lỗi, rồi raise lại.
     - Chỉ tiến trình chính ghi file (dùng `accelerate.PartialState`).
3. **`src/data/pipeline.py`**:
   - Bỏ `logging.basicConfig` và đoạn `try/except` tự ghi `run_info`; thay bằng `start_run("preprocess", ...)`.
   - `prepare_output_dir` giữ nguyên logic báo lỗi khi thư mục đã tồn tại và ghi đè các file được quản lý, nhưng dùng hằng số từ `run.py`.
4. **Config có section `run:`**:
   - `PreprocessConfig`: thay `preprocessing.name` và `output.{directory, overwrite}` bằng `run: {name, output_root, overwrite}`; thư mục output là `output_root/name`.
   - `TrainConfig`: chuyển `trainer.{output_root, run_name}` sang `run: {name, output_root}`.
   - Model chung `RunConfig` đặt trong `src/utils/config.py`. Preprocess dùng thêm trường `overwrite`.
5. **Script theo một khuôn chung**:
   - `scripts/train.py` và `scripts/preprocess.py` bỏ `logging.basicConfig`, vì `start_run` đã lo.
   - `scripts/evaluate.py` gọi `setup_logging(None)`.
   - Kiểm tra: smoke test vẫn chạy với PhoBERT.

## Giai đoạn 2: Chuyển sang fastText (preprocess và model)
Mục tiêu: làm đúng §7. Kết thúc giai đoạn, processed dataset mới và model `naml` mới chạy được smoke test.

**Chuẩn bị:** chép hoặc tạo symlink file `D:\Intern\VCCorp\NRMS_real_data\NRMS_real_data\data\raw\fasttext\cc.vi.300.vec` (4,5GB) vào `data/raw/fasttext/cc.vi.300.vec`. Thư mục `data/` không lên git. Trên Windows có thể dùng `mklink` để khỏi tốn thêm 4,5GB.

**Preprocess:**
1. **`src/data/constants.py`**: thêm `PAD_TOKEN = "<PAD>"`, `UNK_TOKEN = "<UNK>"`, `PAD_TOKEN_IDX = 0`, `UNK_TOKEN_IDX = 1`, `VOCAB_FILE = "vocab.json"`, `EMBEDDING_FILE = "embedding.npy"`.
2. **`src/data/text.py`** (mới):
   - `tokenize(text) -> list[str]`: chuẩn hoá NFKC → chữ thường → regex `\w+|[^\w\s]`.
   - `build_vocab(news_df, text_columns) -> dict[str, int]`: PAD và UNK đứng đầu, các token còn lại sắp theo tần suất giảm dần rồi theo thứ tự chữ cái để kết quả ổn định giữa các lần chạy.
   - `encode(tokens, vocab, max_length) -> np.ndarray`.
3. **`src/data/embedding.py`** (mới): `build_embedding_matrix(vectors_path, vocab, dim, seed, unmatched_init_std)`.
   - Đọc file `.vec` theo dòng (stream), không load cả file vào RAM.
   - Bỏ qua dòng header. Báo lỗi ngay nếu số chiều trong file không khớp `dim`.
   - Thứ tự tra: khớp chính xác trước, rồi thử dạng viết hoa. Từ không tìm thấy thì khởi tạo theo `N(0, std)`. Hàng PAD bằng 0.
   - Trả về ma trận cùng thống kê độ phủ.
   - Tham khảo `NRMS_real_data/src/data/embedding.py`, nhưng bỏ phần hỗ trợ định dạng GloVe.
4. **`src/data/preprocessing.py`**:
   - Viết lại `build_token_matrix` theo vocab: mỗi view một ma trận int32 kích thước `[N+1, L]`, hàng `NEWS_PAD_IDX` toàn `PAD_TOKEN_IDX`.
   - Thống kê độ dài token chuyển sang đếm theo âm tiết.
   - Thêm `_dedup_history(history)`: giữ lần xuất hiện cuối cùng và giữ nguyên thứ tự. Dùng trong `build_train_samples` và `build_eval_samples`.
   - Bỏ `pyvi` và `AutoTokenizer`.
5. **`src/data/config.py`**:
   - Bỏ `TokenizerConfig` và `text.lowercase` (§7 bắt buộc luôn chuyển chữ thường).
   - Thêm `input.word_vectors: FilePath` và `embedding: {dim, unmatched_init_std, seed}`.
6. **`src/data/pipeline.py`**: lưu thêm `vocab.json` và `embedding.npy`. `statistics.json` ghi thêm độ phủ (theo loại từ và theo token), kích thước vocab và số tin trùng đã bỏ trong lịch sử.
7. **`configs/preprocess.yaml`**: đặt `run.name: real_data_fasttext_v1`. Độ dài tối đa tạm để title 32, sapo 64, tags 32; sau khi chạy xem `p95` trong `statistics.json` rồi chỉnh lại.

**Model:**
8. **`src/models/naml.py`**:
   - Xoá `NewsEncoder` (PhoBERT), `freeze_encoder` và `pretrained_parameters`.
   - Chuyển `TextCNNEncoder` và `CNNNewsEncoder` từ `naml_cnn.py` sang, đổi tên `CNNNewsEncoder` thành `NewsEncoder`.
   - `pad_token_id` lấy từ `PAD_TOKEN_IDX` trong constants.
   - Xoá file `src/models/naml_cnn.py`.
9. **`src/models/config.py`**: chỉ còn một `NAMLConfig` với `name: Literal["naml"]`, `freeze_embedding`, `num_filters`, `window_size`, `category_dim`, `query_dim`, `dropout`. Số chiều embedding lấy từ ma trận, không khai báo trong config.
10. **`src/models/registry.py`**: chỉ còn `build_naml(config, num_categories, text_views, word_embedding)`; bỏ `AutoModel`.
11. **`src/data/news.py`**: `NewsFeatures` load thêm `embedding.npy` (thuộc tính `embedding`) và bỏ tham số `pretrained_name` cùng bước kiểm tra tokenizer.
12. **`src/engine/config.py`**:
    - `model: NAMLConfig`, bỏ union.
    - `optimizer: {lr, weight_decay}`, bỏ `encoder_lr` vì embedding dùng chung learning rate như NRMS.
    - Bỏ config `naml_cnn` và `train_naml_cnn.yaml`.
13. **Kiểm tra:**
    - Chạy preprocess đầy đủ (đọc file `.vec` mất khoảng 3–5 phút); độ phủ token phải xấp xỉ 99,7%.
    - Chạy smoke test.
    - Gỡ `transformers` và `pyvi` khỏi `requirement.txt`.
    - Xoá `data/processed/real_data_v1` cũ, sau khi bạn đồng ý.

## Giai đoạn 3: Vòng train (§8, §11)
Mục tiêu: early stopping, scheduler, gradient clipping, chỉ giữ checkpoint `best`/`latest`, có `history.json` và biểu đồ, log mỗi epoch một dòng, có tqdm. Resume vẫn khớp chính xác.

1. **`src/engine/config.py`**: thêm các section và trường sau.
   - `trainer`: thêm `max_grad_norm`, `monitor: Literal["auc", "impression_auc", "mrr", "ndcg@5", "ndcg@10"]`; bỏ `log_every`.
   - `scheduler: {factor, patience, min_lr}`.
   - `early_stopping: {patience, min_delta}`.
   - `data.max_train_samples: PositiveInt | None`: dùng cho smoke test để có nhiều epoch rất ngắn.
2. **`src/engine/trainer.py`**:
   - **Khởi tạo**: gọi `start_run("train", run_dir, config, resume=...)`. Bỏ `log_with="tensorboard"` và `init_trackers`.
   - **Scheduler**: `torch.optim.lr_scheduler.ReduceLROnPlateau(mode="max")`. Không đưa qua `prepare`, mà đăng ký bằng `accelerator.register_for_checkpointing(scheduler)`. Gọi `scheduler.step(val[monitor])` sau mỗi epoch.
   - **Gradient clipping**: `accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)` khi `accelerator.sync_gradients`.
   - **Trạng thái train**: `training_state.json` lưu trong checkpoint, gồm `epoch`, `batches_in_epoch`, `global_step`, `loss_sum`, `loss_count` (để loss trung bình của epoch đang dở vẫn khớp sau khi resume), `best_value`, `best_epoch`, `bad_epochs`, `history`.
   - **Checkpoint**:
     - `latest/`: lưu mỗi `save_every_steps` và cuối mỗi epoch.
     - `best/`: lưu khi metric tốt hơn `best_value + min_delta`, kèm `metrics.json`.
     - Mỗi checkpoint có `config.yaml` riêng.
     - Bỏ `epoch_<i>` và `EPOCH_CHECKPOINT_TEMPLATE`.
   - **Early stopping**: dừng khi `bad_epochs >= patience`. Sau đó log lý do và dòng tổng kết epoch tốt nhất.
   - **Log**: mỗi epoch đúng một dòng: `epoch=… loss=… auc=… imp_auc=… mrr=… ndcg@5=… ndcg@10=… lr=… time=… [best]`. Đầu run log tên run, device, số mẫu, số tham số, độ phủ embedding (đọc từ `statistics.json`). Cuối run log tổng thời gian.
   - **tqdm**:
     - Thanh train hiển thị `loss`/`lr` dưới dạng postfix.
     - Đặt `leave=False`; `disable=None` ở tiến trình chính (tự tắt khi output không phải terminal), `True` ở các tiến trình khác.
     - Bọc vòng train bằng `logging_redirect_tqdm`.
   - **Giữ nguyên** `fork_rng` quanh bước evaluate, RNG cố định theo (seed, epoch), `skip_first_batches` và `_check_resume_config` (cập nhật danh sách trường cố định).
3. **`src/engine/plots.py`** (mới, chỉ trainer dùng nên đặt trong engine): `plot_history(history, plots_dir)` dùng backend `Agg`.
   - `loss.png`: train loss theo epoch.
   - `metrics.png`: 5 metric val, đánh dấu epoch tốt nhất.
   - `lr.png`: learning rate theo epoch.
   - Gọi sau mỗi epoch, cùng lúc ghi `history.json` ở gốc thư mục run.
4. **`src/engine/evaluate.py`** (hàm `evaluate`): thêm thanh tqdm cho bước encode tin và bước chấm điểm impression.
5. **Configs**:
   - `train.yaml`: batch 256, `lr` 2e-4, `max_grad_norm` 5.0, scheduler `factor` 0.5 / `patience` 1 / `min_lr` 1e-6, early stopping `patience` 3 / `min_delta` 1e-4, 30 epoch, `bf16`, `save_every_steps` 500.
   - `train_smoke.yaml`: `max_train_samples: 64`, batch 8, 3 epoch, `early_stopping.patience: 1`, `max_eval_impressions: 20`.

## Giai đoạn 4: Evaluate (§10)
1. **`run_evaluation(checkpoint_dir)`**:
   - Đọc `checkpoint_dir/config.yaml`.
   - Thư mục output là `<run_dir>/evaluation_<tên checkpoint>/`.
   - Chỉ chạy tập test. Bỏ tham số `split`.
2. **`evaluate(..., collect_predictions)`**: nếu bật thì trả thêm điểm của từng candidate kèm vị trí impression và row của tin.
   `run_evaluation` ghép với `impression_id`, `user_id` và `newsId`, tính `rank` trong từng impression (groupby rồi xếp hạng giảm dần), sau đó ghi `predictions.csv` bằng pandas.
3. **`scripts/evaluate.py`**: chỉ nhận `--checkpoint`, mặc định gợi ý `checkpoints/best`; gọi `setup_logging(None)`.
4. **Kiểm tra**: chạy evaluate trên checkpoint `best` của smoke test. Có `metrics.json` và `predictions.csv`, số dòng của CSV bằng tổng số candidate.

## Giai đoạn 5: Tài liệu và dọn dẹp
1. **README**:
   - Bỏ các phần về PhoBERT, HuggingFace và tensorboard. Cài đặt môi trường không còn cần tải PhoBERT.
   - Thêm bước chuẩn bị file `.vec`.
   - Cập nhật cấu trúc output, mẫu log, cách xem biểu đồ, và evaluate chỉ chạy trên test.
2. **Rà lại toàn bộ code theo CLAUDE.md**:
   - Không còn chuỗi tên file hay thư mục viết rải rác.
   - Không còn trường config nào thừa.
   - Không còn import thừa (`ruff check --select F`).
3. Commit sau **mỗi** giai đoạn, sau khi smoke test chạy qua.

---

## File chính bị ảnh hưởng
- **Mới:** `src/utils/run.py`, `src/data/text.py`, `src/data/embedding.py`, `src/engine/plots.py`
- **Xoá:** `src/models/naml_cnn.py`, `configs/train_naml_cnn.yaml`
- **Sửa nhiều:** `src/data/{pipeline,preprocessing,config,constants,news}.py`, `src/models/{naml,config,registry}.py`, `src/engine/{trainer,evaluate,config}.py`, `scripts/*.py`, `configs/*.yaml`, `requirement.txt`, `README.md`
- **Dùng lại, không viết lại:**
  - Khung train: `TrainDataset` với RNG theo (seed, epoch), `TrainCollator` (encode mỗi tin một lần trong batch), cách tính metric bằng torchmetrics trong `evaluate()`, `fork_rng`, `skip_first_batches`, `_check_resume_config`.
  - Tiện ích: `load_config`/`save_config`, `save_json`/`load_json`.

## Kiểm tra end-to-end
1. `python -m scripts.preprocess --config configs/preprocess.yaml`: kiểm tra `statistics.json` có độ phủ xấp xỉ 99,7% và số tin trùng đã bỏ.
2. `python -m scripts.train --config configs/train_smoke.yaml`:
   - Mỗi epoch log đúng một dòng.
   - Early stopping kích hoạt (`patience` 1).
   - Có `history.json`, `plots/*.png`, `checkpoints/{best,latest}`.
   - `run_info.json` có `status: completed`.
3. **Resume khớp chính xác** theo hai kịch bản:
   - **Giữa epoch:** chạy liên tục N step, so với chạy `max_train_steps` = N/2 rồi `--resume checkpoints/latest`. Loss từng step và metric val phải trùng nhau.
   - **Ranh giới epoch:** chạy 2 epoch, resume lên 3 epoch, so với chạy liên tục 3 epoch.
4. **Báo lỗi đúng cách:**
   - Resume với config model khác thì phải báo lỗi.
   - Chạy preprocess khi thư mục đã tồn tại thì phải báo lỗi.
   - Ép một exception giữa chừng thì `run_info.json` phải ghi `failed`.
5. `python -m scripts.evaluate --checkpoint outputs/<run>/checkpoints/best`: có `evaluation_best/metrics.json` và `predictions.csv`.
6. Train thật `configs/train.yaml` trên máy thuê: theo dõi log và biểu đồ, rồi evaluate checkpoint `best`.
