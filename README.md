# MNIST BPNN：逐階段從零實作

Phase 1 紀錄：固定 `784 → 128 → 10` 架構，以 NumPy 手刻前向傳播、交叉熵、反向傳播與逐筆 SGD 更新。沒有使用神經網路框架或自動微分。

## 執行 Phase 1

需要 Python 3 與 NumPy：

```bash
python3 phase1_baseline.py
```

程式使用一筆合成的 `(1, 784)` 輸入展示計算，並檢查各層形狀，以及 `W1`、`W2` 各一個元素的數值梯度。這是演算法核心範例；尚未接上 MNIST PNG 資料載入、完整訓練或助教指定的預測文字檔。

## Phase 2：通用多層架構

`phase2_multilayer.py` 將固定隱藏層改為 `hidden_layers` tuple，例如 `(256, 128)` 對應 `784 → 256 → 128 → 10`。將此 tuple 改成 `(128,)` 或 `(64, 32, 16)`，其餘前向與反向程式碼都不需改動。`phase1_baseline.py` 仍保留作為單隱藏層、逐步展開公式的紀錄；Phase 2 延續逐筆 SGD、ReLU、Softmax 與交叉熵。

```bash
python3 phase2_multilayer.py
```

前向傳播用迴圈依序計算每層的 `z = a @ W + b` 與活化值，並保存它們供反向傳播使用。若第 `i` 層輸入寬度為 `n_i`、輸出寬度為 `n_{i+1}`，則 `(1, n_i) @ (n_i, n_{i+1})` 得到 `(1, n_{i+1})`。輸出層從 `delta = p - y` 開始；反向迴圈先以 `a.T @ delta` 算該層權重梯度，再以 `delta @ W.T` 把梯度傳回前一層，並乘上該層 ReLU 導數 `(z > 0)`。所有梯度算完後才更新參數，避免使用到已更新的權重。

執行範例仍使用一筆合成輸入，檢查各層形狀及代表性權重的數值梯度；它不是 MNIST 完整訓練程式。Phase 2 的重點是理解迴圈如何對應 Phase 1 逐層寫出的鏈鎖律。

## Phase 2 延伸：mini-batch 與 epoch 訓練

`phase2_train.py` 將 Phase 2 的多層前向、反向迴圈套用到真實 MNIST。每個 epoch 先打亂 60,000 筆訓練資料，按 batch 更新參數；最後不足一個 batch 的樣本也會訓練。保留 one-hot 標籤、基本 Softmax 和交叉熵，讓 Phase 3 再處理標籤記憶體用量與數值穩定性。

這裡使用 [CVDF Foundation 提供的 MNIST 鏡像](https://github.com/cvdfoundation/mnist/blob/master/README.md)，下載原始 IDX gzip 格式的圖片和標籤。下載檔只放在本機的 `data/mnist/`，Git 不會追蹤 `data/`：

```bash
mkdir -p data/mnist
for name in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -fL "https://storage.googleapis.com/cvdf-datasets/mnist/$name" -o "data/mnist/$name"
done
```

需要 Python 3 與 NumPy；從專案根目錄執行：

```bash
OPENBLAS_NUM_THREADS=1 python3 phase2_train.py --data-dir data/mnist --epochs 5 --batch-size 64 --learning-rate 0.05 --hidden-layers 256 128
```

若 batch 大小為 `B`，輸入 `X` 是 `(B, 784)`，one-hot 標籤 `Y` 與輸出 `P` 是 `(B, 10)`。平均交叉熵的輸出梯度為 `delta = (P - Y) / B`；每層的 `dW = A.T @ delta`，偏置梯度 `db = sum(delta, axis=0, keepdims=True)`。`delta` 保留每筆樣本的梯度，`dW`、`db` 則把它們彙整成一次參數更新；所有層的梯度都先算完，再一起更新。訓練完會用 MNIST 的 10,000 筆測試集計算準確率。

以這組設定和亂數種子 `7` 實際訓練的結果如下。訓練指標是在每個 epoch 更新參數的過程中累計，並非用該 epoch 最後的參數重新計算。

| epoch | 訓練平均 loss | 訓練準確率 |
|---:|---:|---:|
| 1 | 0.3917 | 89.05% |
| 2 | 0.1900 | 94.51% |
| 3 | 0.1420 | 95.88% |
| 4 | 0.1136 | 96.72% |
| 5 | 0.0948 | 97.27% |

最後在 MNIST 測試集的準確率為 **96.98%**。

作業現場的資料格式是 28×28 PNG，與這裡的 IDX 不同；`phase2_train.py` 當時尚未包含 PNG 載入及指定格式的預測輸出，下節的課堂範圍版才補上。`phase1_baseline.py` 和 `phase2_multilayer.py` 保留原本的逐步學習紀錄。

## 課堂範圍版：PNG 訓練、驗證與預測輸出

`coursework_mlp.py` 將目前要用的功能接在一起，仍由 NumPy 手刻網路與反向傳播，Pillow 只負責讀取 PNG。這個版本**保留 one-hot** 與 `delta = (P - Y) / B`；原先規劃的整數標籤梯度技巧、Adam 和 L2 不在目前課堂範圍內。前面各階段的檔案與實驗數據留作學習紀錄。

作業 PDF 沒有指定有標籤訓練圖片的存放方式。這裡採用目前約定的數字子資料夾；圖片須為 28×28，讀入後轉灰階、展平成 784 個像素，再除以 255：

```text
train/
  0/00001.png
  1/00002.png
  ...
  9/00009.png
test/
  0000.png
  0001.png
  0x00.png
```

如果要用先前下載的 IDX 版 MNIST 重現這個 PNG 流程，可用 `prepare_png_mnist.py` 轉換。第一個指令會轉全部 60,000 張訓練圖與 10,000 張測試圖；第二個指令是可選的小樣本練習，使用另一個輸出資料夾，避免留下不同筆數的舊檔。`data/` 已由 Git 忽略：

```bash
python3 prepare_png_mnist.py
python3 prepare_png_mnist.py --output-dir data/mnist_png_sample --train-count 3000 --test-count 21
```

需要 Python 3、NumPy 與 Pillow。直接執行時預設讀取本專案的 `data/mnist_png/train`，完成 5 個 epoch 並印出訓練與驗證指標；**無參數執行不會產生預測檔**：

```bash
python3 coursework_mlp.py
```

要在同一次訓練後讀取測試圖片並輸出學號文字檔，請把 `A12345678` 換成自己的學號：

```bash
python3 coursework_mlp.py --predict-dir data/mnist_png/test --student-id A12345678
```

若使用其他有標籤資料集，再加 `--train-dir 路徑`；`--epochs`、`--batch-size`、`--learning-rate`、`--anneal-factor`、`--hidden-layers` 可調整實驗設定。

若沒有 `--val-dir`，程式在訓練前固定留出 10% 有標籤圖片作 validation；若已有獨立的數字子資料夾驗證集，可傳入 `--val-dir`。每個 epoch 會重新 shuffle 訓練資料、切 mini-batch，並用 `learning_rate × anneal_factor^(epoch-1)` 遞減學習率。每輪更新完畢後，重新計算並印出 Train Loss、Train Accuracy、Validation Accuracy；最後以當輪訓練出的權重讀取 `--predict-dir` 的全部 PNG，按檔名排序，寫出無表頭的 `A12345678.txt`，每行是「不含 `.png` 的檔名、Tab、預測數字」。

輸出層先對每張圖片的 logits 減去該列最大值，再計算 Softmax，避免 `exp` 溢位。平均交叉熵使用 `-sum(Y * log(P + 1e-12)) / B`，避免 `log(0)`；反向傳播仍使用課堂上的 `delta = (P - Y) / B`。`P-Y` 是未加平滑的標準交叉熵梯度；加上 `1e-12` 後，兩者在一般機率下幾乎相同，但在正確類別機率極接近 0 時會有明顯差異。

完整資料量的實測使用 60,000 張 MNIST 圖片轉成 PNG，固定留出 6,000 張 validation；參數為 `784 → 128 → 10`、batch size `64`、初始學習率 `0.05`、anneal factor `0.95`、seed `7`。以下指標均在各 epoch 更新結束後重算：

| epoch | 學習率 | Train Loss | Train Accuracy | Validation Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.050000 | 0.3094 | 90.88% | 90.62% |
| 2 | 0.047500 | 0.2372 | 93.38% | 93.43% |
| 3 | 0.045125 | 0.1976 | 94.51% | 94.33% |
| 4 | 0.042869 | 0.1744 | 95.18% | 94.82% |
| 5 | 0.040725 | 0.1559 | 95.73% | 95.18% |

另外，這次從 MNIST 測試集中取前 100 張轉為無標籤 PNG，產生了 100 行預測，與原標籤比對有 98 張正確；此 100 張結果只是輸出流程檢查，並非完整測試集準確率。若助教現場提供不同的標籤格式，需依實際格式調整 `load_labeled_pngs()`；目前資料夾規則是我們的約定，並非 PDF 指定。
