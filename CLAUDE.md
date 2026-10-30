# AI Training Project Guidelines

## 0. Phạm vi
- `create_data/` (notebook, notes, raw_log, pipeline tạo data từ log thô) nằm NGOÀI phạm vi các quy tắc dưới đây. KHÔNG sửa, refactor hay "dọn dẹp" thư mục này nếu không được yêu cầu rõ ràng.
- Không đọc, in ra hay commit `.env`. Không load toàn bộ `create_data/raw_log/` vào RAM.

## 1. Kiến trúc & Phân tách trách nhiệm (Strict Modularity)
- TUYỆT ĐỐI KHÔNG viết toàn bộ Dataset, Model, và Training Loop vào chung một file.
- Cấu trúc bắt buộc:
  - `src/data/`: tiền xử lý (`data/raw/` → `data/processed/`), Dataset, DataCollator, hằng số domain.
  - `src/models/`: Kiến trúc mạng thuần túy (`nn.Module`). Không chứa logic optimizer hay đường dẫn file.
  - `src/models/registry.py`: Registry DUY NHẤT của project. Không tạo registry/factory thứ hai.
  - `src/engine/`: Logic bước train/eval hoặc LightningModule / Trainer subclass.
  - `src/utils/`: Chỉ chứa tiện ích dùng chung cho ≥ 2 module (config loader, I/O, khởi tạo run — xem §10). Code chỉ phục vụ một module thì đặt trong module đó.
  - `configs/`: Mọi hyperparameter phải nằm ở đây, không hardcode trong code.
  - `scripts/`: Chỉ chứa entry-point mỏng (dưới 80 dòng) để đọc config và gọi engine.

## 2. Cấm tự viết lại Boilerplate (Don't Reinvent the Wheel)
- KHÔNG tự viết vòng lặp training thủ công xử lý `.to(device)`, `GradScaler`, `DistributedDataParallel` (DDP), hay gradient accumulation. Bắt buộc dùng `accelerate` (HuggingFace). Device, AMP, gradient accumulation do `Accelerator` quản lý.
- KHÔNG tự viết hàm tính metric. Bắt buộc dùng `torchmetrics` (ranking: `BinaryAUROC`, `RetrievalAUROC`, `RetrievalMRR`, `RetrievalNormalizedDCG`).
- KHÔNG tự viết code quản lý lưu/xóa checkpoint thủ công bằng `os.makedirs` và `torch.save` rải rác. Dùng `accelerator.save_state` / `load_state`.
- KHÔNG dùng các phép biến đổi tensor phức tạp kiểu `.view().permute().transpose()` khó đọc. Dùng `einops` (`rearrange`, `reduce`, `repeat`).
- Thanh tiến trình dùng `tqdm`, không tự in tiến độ bằng `print`/`logging`.

## 3. Quy tắc Tensor & Type Safety
- Mọi dòng làm THAY ĐỔI shape của Tensor PHẢI có comment chú thích shape ngay trên dòng đó (ví dụ: `# x: [B, S, D] -> [B, D]`). Hàm nhận/trả Tensor ghi shape trong docstring.
- Config định nghĩa bằng `Pydantic` (`BaseModel`, `extra="forbid"`) có type-hint rõ ràng — không trộn với `dataclass`. Cấm truyền dictionary tự do (`dict[str, Any]`) hoặc dùng `argparse` với hàng chục cờ rời rạc.
- Mỗi trường trong config phải được code sử dụng. Không thêm trường "để sau dùng".

## 4. Nguyên tắc Fail-Fast (Cấm Defensive Coding mù quáng)
- TUYỆT ĐỐI KHÔNG bọc `try...except` quanh bước đọc dữ liệu (`__getitem__`), `forward()`, hoặc `loss.backward()` để bỏ qua lỗi. Lỗi dữ liệu hoặc NaN loss phải làm chương trình dừng ngay lập tức (crash early).
- Không viết các hàm fallback tự động đoán đường dẫn file hoặc tự sinh dummy data nếu không được yêu cầu rõ ràng.

## 5. Chống trùng lặp & code thừa
- Trước khi viết hàm/class mới, PHẢI tìm trong `src/` xem đã có cái tương tự chưa. Có rồi thì dùng lại hoặc mở rộng, không viết bản thứ hai.
- Hằng số domain (tên cột, mapping `eventId`, `PAD_IDX`, token đặc biệt) chỉ định nghĩa ở MỘT chỗ: `src/data/constants.py`.
- Code không dùng nữa thì XÓA, không comment-out (đã có git). Xóa luôn import thừa.
- Không tạo file kiểu `*_v2.py`, `*_new.py`, `*_old.py`, không viết shim tương thích ngược.
- Chỉ sửa đúng phần được yêu cầu, không tiện tay refactor file không liên quan.
- Không thêm dependency mới khi chưa hỏi. Danh sách thư viện được phép nằm trong `requirement.txt`.

## 6. Quy tắc khi thêm thử nghiệm mới (Experimentation)
- Khi thêm kiến trúc model mới hoặc hàm loss mới, KHÔNG được nhân bản file `train.py`. Viết module mới và đăng ký vào `src/models/registry.py` để chọn qua file config.
- Giữ code tối giản (YAGNI): chỉ thêm tính năng khi có nhu cầu thật.

## 7. Biểu diễn văn bản (tiếng Việt)
- Word embedding dùng fastText `cc.vi.300.vec` (300 chiều). KHÔNG dùng PhoBERT/transformer cho news encoder.
- Tách token đúng một cách duy nhất, dùng chung cho mọi view văn bản: chuẩn hoá Unicode NFKC → chữ thường → regex `\w+|[^\w\s]` (tách theo âm tiết, mỗi dấu câu là một token). KHÔNG dùng ViTokenizer/tách từ ghép: fastText `cc.vi` không có vector cho từ ghép (`học_sinh`).
- Từ điển dựng từ text của news; `<PAD>` và `<UNK>` định nghĩa trong `src/data/constants.py`.
- Ma trận embedding dựng MỘT lần lúc preprocess (khớp chính xác trước, sau đó thử dạng viết hoa, cuối cùng khởi tạo ngẫu nhiên; hàng `<PAD>` = 0) và lưu thành artifact `embedding.npy`. Train không đọc file `.vec`. Độ phủ (coverage) ghi vào `statistics.json`.
- Lịch sử click được bỏ tin trùng (giữ lần click gần nhất) trước khi cắt `history_len`.

## 8. Quy tắc train
- Optimizer AdamW (`weight_decay` trong config), gradient clipping qua `accelerator.clip_grad_norm_`.
- Scheduler `ReduceLROnPlateau` theo metric val, step một lần mỗi epoch; trạng thái scheduler phải được lưu cùng checkpoint (`accelerator.register_for_checkpointing`).
- Early stopping theo `monitor` (mặc định `ndcg@10`, cùng metric với scheduler), `patience`, `min_delta`. Giá trị tốt nhất và bộ đếm patience lưu cùng checkpoint để resume không bị reset.
- Checkpoint chỉ gồm `checkpoints/best/` (tốt nhất theo `monitor`, kèm `metrics.json`) và `checkpoints/latest/` (ghi đè định kỳ để resume). Không giữ checkpoint theo từng epoch.
- Resume phải cho kết quả giống chạy liên tục: thứ tự shuffle, negative sampling và RNG cố định theo (seed, epoch); evaluate trong lúc train không được làm lệch RNG.
- Val chỉ đánh giá trong lúc train (sau mỗi epoch). Tập test KHÔNG được dùng trong lúc train; chỉ đánh giá bằng `scripts/evaluate.py` trên checkpoint đã chọn (mặc định `best`).

## 9. Đúng đắn của bài toán Recommendation
- Chia train/val/test theo thời gian. Lịch sử đọc của user chỉ lấy từ trước thời điểm impression.
- Negative sample chỉ lấy trong cùng impression.
- Cố định seed cho mọi nguồn ngẫu nhiên.

## 10. Quy chuẩn run
- Mỗi lần chạy một script là một run. Config nào cũng có section `run: {name, output_root}`.
- Thư mục run:
  - Dữ liệu processed: `output_root/name` (tên cố định vì config train trỏ tới nó); đã tồn tại thì báo lỗi, trừ khi bật `overwrite`.
  - Thí nghiệm train: `output_root/<name>_<timestamp>`; resume ghi tiếp vào thư mục cũ.
  - Evaluate (chỉ tập test): lưu trong thư mục run train, `evaluation_<checkpoint>/`, chỉ gồm `metrics.json` và `predictions.csv` (mỗi dòng một cặp impression–candidate: `impression_id`, `user_id`, `news_id`, `label`, `score`, `rank`).
- Run preprocess và train có cùng các file:
  - `config.yaml`: config đã resolve, KHÔNG bao giờ bị ghi đè. Mỗi checkpoint cũng mang `config.yaml` của nó.
  - `run_info.json`: stage, trạng thái (`running`/`completed`/`failed`), thời gian, phiên bản python/torch/CUDA, GPU, git commit, câu lệnh; mỗi lần resume thêm một mục, không ghi đè.
  - `<stage>.log`: log ghi ra cả file và màn hình (xem §11).
- Evaluate là ngoại lệ: không có `config.yaml`, `run_info.json` hay file log riêng (config đọc từ checkpoint, log chỉ in ra màn hình).
- Phần chung (tạo thư mục, cấu hình logging, lưu config, ghi `run_info.json` kể cả khi lỗi rồi raise lại) nằm trong MỘT hàm dùng chung ở `src/utils/`; preprocess và train gọi hàm này, không tự làm lại. Evaluate dùng chung phần cấu hình logging của hàm đó.
- Script nào cũng theo một khuôn: đọc tham số tối thiểu (`--config`, thêm `--resume` cho train; evaluate chỉ nhận `--checkpoint`), validate config bằng Pydantic, gọi đúng một hàm trong `src/`.
- Tên file/thư mục cố định của run layout định nghĩa ở MỘT chỗ, không viết chuỗi rải rác.

## 11. Log & theo dõi kết quả
- KHÔNG dùng `tensorboard`/`wandb`.
- Log text dùng thẳng `logging` chuẩn: mỗi module `logger = logging.getLogger(__name__)`. Logging chỉ được cấu hình MỘT lần, trong hàm khởi tạo run dùng chung (§10) — không gọi `logging.basicConfig` ở chỗ khác. KHÔNG viết module/hàm wrapper cho logger.
- Log chỉ ghi các mốc quan trọng, KHÔNG log từng step:
  - Đầu run: tên run và thư mục, device/mixed precision, số mẫu, số tham số (và coverage embedding nếu có).
  - Mỗi epoch đúng MỘT dòng: `epoch=<i> loss=<loss trung bình epoch> <các metric val> lr=<lr> time=<thời gian epoch>`, thêm ` best` nếu là epoch tốt nhất.
  - Các sự kiện: resume (từ đâu, epoch/step nào), early stopping (lý do), tổng kết epoch tốt nhất và tổng thời gian.
- Tiến trình theo step (train, encode news, chấm điểm val/test) chỉ hiển thị bằng `tqdm` trên terminal: `leave=False`, chỉ ở main process, tự tắt khi output không phải TTY, dùng `logging_redirect_tqdm` để không vỡ thanh. Thanh tiến trình KHÔNG được ghi vào file log.
- Metric/loss mỗi epoch ghi vào `history.json` của run (`epoch`, `train_loss`, `lr`, `time`, các metric val); biểu đồ vẽ bằng `matplotlib` từ `history.json` (chỉ ở đây, không vẽ ở chỗ khác), lưu vào `plots/` của run và vẽ lại sau mỗi epoch.

## 12. Kiểm tra sau khi sửa
- Sau mỗi thay đổi ở `src/` hoặc `configs/`, chạy smoke test: `python -m scripts.train --config configs/train_smoke.yaml`.
