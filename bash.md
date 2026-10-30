## Các lệnh theo thứ tự

- Clone repo:
    ```bash
    git clone https://github.com/dinhtrieuxtnd/NAML.git
    cd NAML
    ```
- Cài python 3.12 nếu chưa có hoặc chưa phải là 3.12:
    ```bash
    sudo apt install python3.12 python3.12-venv
    ```
- Tạo và kích hoạt môi trường ảo với python 3.12 (mọi lệnh `pip` sau đều chạy trong môi trường này):
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
    ```
- Cài pytorch (đổi `cu126` theo `CUDA Version` của `nvidia-smi`, xem README):
    ```bash
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
    ```
- Cài các thư viện khác:
    ```bash
    pip install -r requirement.txt
    ```
- Tải dữ liệu:
    ```bash
    pip install gdown
    gdown "https://drive.google.com/file/d/1xldSpS63doRgA94rRoSjFgY-Iwy2CMpJ/view?usp=sharing"
    ```
- Tải unrar:
    ```bash
    sudo apt install unrar
    ```
- Giải nén dữ liệu (phải ra `data/processed/real_data_fasttext_v3/`):
    ```bash
    unrar x data.rar data/processed
    ```
- Chạy smoke test:
    ```bash
    python -m scripts.train --config configs/train_smoke.yaml
    ```
- Chạy train:
    ```bash
    python -m scripts.train --config configs/train.yaml
    ```
- Chạy eval:
    ```bash
    python -m scripts.evaluate --checkpoint outputs/<run>/checkpoints/best
    ```
