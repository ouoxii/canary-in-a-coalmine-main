# Canary in a Coalmine - 完整執行指南

本文檔提供了從環境創建到執行完整攻擊的逐步操作指南。

## 目錄
1. [環境設置](#環境設置)
2. [依賴安裝](#依賴安裝)
3. [修復 Windows 兼容性問題](#修復-windows-兼容性問題)
4. [準備模型](#準備模型)
5. [執行攻擊](#執行攻擊)
6. [查看結果](#查看結果)

---

## 環境設置

### 步驟 1：創建 Conda 環境

```powershell
conda create -n canary python=3.10 -y
```

### 步驟 2：激活環境

```powershell
conda activate canary
```

---

## 依賴安裝

### 步驟 1：安裝 PyTorch（CUDA 版本）

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果不使用 GPU，可改用 CPU 版本：
```powershell
pip install torch torchvision torchaudio
```

### 步驟 2：安裝核心依賴

```powershell
pip install scikit-learn wandb pynvml einops timm tensorboard
```

**依賴說明：**
- `scikit-learn` - 機器學習工具（用於計算 AUC、ROC 曲線）
- `wandb` - 實驗監控（可選，使用 `--nowandb` 禁用）
- `pynvml` - GPU 監控
- `einops` - 張量操作（Vision Transformer 需要）
- `timm` - 預訓練模型庫
- `tensorboard` - 訓練可視化

---

## 修復 Windows 兼容性問題

### 問題 1：多進程 DataLoader 崩潰

**症狀**：
```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

**解決方案**：
修改 [train.py](train.py) 第 52 行，將所有執行代碼包裹在 `if __name__ == '__main__':` 下：

```python
if __name__ == '__main__':
    # 所有訓練代碼都應在此縮進一級
    ...
```

**狀態**：已在項目中修復 ✅

### 問題 2：PyTorch 2.6+ 加載模型失敗

**症狀**：
```
_pickle.UnpicklingError: Weights only load failed
```

**解決方案**：
修改 [models/inferencemodel.py](models/inferencemodel.py) 第 27 行：

```python
# 改為
checkpoint = torch.load(resume_checkpoint, weights_only=False)
```

**狀態**：已在項目中修復 ✅

---

## 準備模型

### 選項 A：下載預訓練模型（推薦 ⭐）

**優勢**：
- ✅ 快速（無需訓練）
- ✅ 使用官方模型
- ✅ 可立即運行攻擊

**步驟**：

1. 訪問 [Google Drive](https://drive.google.com/drive/folders/15aoIRU7rq4P4FVCxHdWV2UwJxQ7xt7rb?usp=sharing)

2. 下載 `wrn28-10` 文件夾（包含 65 個模型）

3. 解壓到項目 `saved_models/` 目錄：
```
saved_models/
└── wrn28-10/
    ├── wrn28-10_target_last.pth      (目標模型)
    ├── wrn28-10_shadow_0_last.pth    (影子模型 0)
    ├── wrn28-10_shadow_1_last.pth    (影子模型 1)
    └── ... (共 64 個影子模型)
```

### 選項 B：訓練影子模型（可選）

**注意**：訓練 64 個模型需要大量時間和 GPU 資源。

**快速測試版本**（4 個影子模型，各 10 epochs）：

```powershell
conda activate canary
python train.py --name wrn28-10 --net wrn28-10 --bs 256 --lr 0.1 --pkeep 0.5 --num_shadow 4 --shadow_id 0 --n_epochs 10 --nowandb
python train.py --name wrn28-10 --net wrn28-10 --bs 256 --lr 0.1 --pkeep 0.5 --num_shadow 4 --shadow_id 1 --n_epochs 10 --nowandb
python train.py --name wrn28-10 --net wrn28-10 --bs 256 --lr 0.1 --pkeep 0.5 --num_shadow 4 --shadow_id 2 --n_epochs 10 --nowandb
python train.py --name wrn28-10 --net wrn28-10 --bs 256 --lr 0.1 --pkeep 0.5 --num_shadow 4 --shadow_id 3 --n_epochs 10 --nowandb
```

然後修改攻擊命令中的 `--num_shadow 4`。

---

## 執行攻擊

### 快速測試（前 10 個樣本）

用於驗證環境和模型是否正確設置：

**Canary Online 版本**：
```powershell
conda activate canary
python gen_canary.py --name wrn28-10 --save_name wrn28-10_online_test --num_shadow 64 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 10 --nowandb
```

**Canary Offline 版本**：
```powershell
python gen_canary.py --name wrn28-10 --save_name wrn28-10_offline_test --offline --num_shadow 64 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adam --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --stop_loss 23 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 10 --nowandb
```

**預期結果**（示例，前 10 個樣本）：
```
Online:
  - fix_auc: 0.583
  - fix_acc: 75%
  - fix_TPR@0.01FPR: 50%

Offline:
  - fix_off_auc: 0.667
  - fix_off_acc: 70.8%
  - fix_off_TPR@0.01FPR: 25%
```

### 完整測試（5000 個樣本）

運行完整攻擊以復現論文結果（耗時較長）：

**Canary Online 版本**：
```powershell
python gen_canary.py --name wrn28-10 --save_name wrn28-10_online_full --num_shadow 64 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000 --nowandb
```

**Canary Offline 版本**：
```powershell
python gen_canary.py --name wrn28-10 --save_name wrn28-10_offline_full --offline --num_shadow 64 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adam --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --stop_loss 23 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000 --nowandb
```

**LiRA 基線**（用於對比）：
```powershell
python gen_canary.py --name wrn28-10 --save_name wrn28-10_baseline --aug_strategy baseline --num_shadow 64 --num_aug 10 --start 0 --end 5000 --nowandb
```

---

## 查看結果

### 終端輸出

攻擊完成時會在終端輸出評估指標：

```python
{
    'fix_auc': 0.9234,           # Online AUC
    'fix_acc': 0.92,             # Online 準確率
    'fix_TPR@0.01FPR': 0.75,     # Online TPR@0.01FPR
    'fix_off_auc': 0.8956,       # Offline AUC
    'fix_off_acc': 0.88,         # Offline 準確率
    'fix_off_TPR@0.01FPR': 0.65  # Offline TPR@0.01FPR
}
```

### 評估指標說明

| 指標 | 含義 | 範圍 | 解釋 |
|------|------|------|------|
| **AUC** | ROC 曲線下面積 | [0, 1] | 越接近 1 越好；0.5 表示隨機猜測 |
| **Accuracy** | 分類準確率 | [0, 1] | 正確預測的比例 |
| **TPR@0.01FPR** | 真陽性率 @1% 誤報率 | [0, 1] | 在控制誤報率 1% 的前提下的真陽性率 |

**論文報告的典型結果**（CIFAR-10, WideResNet-28-10）：
- **Canary Online**: AUC ≈ 0.92-0.95
- **Canary Offline**: AUC ≈ 0.88-0.91
- **LiRA 基線**: AUC ≈ 0.75-0.80

### 保存的文件

攻擊完成後會在以下位置保存結果：

```
saved_predictions/
└── wrn28-10/
    └── wrn28-10_offline_full.npz  # 包含預測和評估指標
```

---

## 常見問題

### Q1：訓練需要多長時間？

| 任務 | 時間 | GPU 配置 |
|------|------|---------|
| 快速測試（10 個樣本） | ~2-5 分鐘 | RTX 3060+ |
| 完整測試（5000 個樣本） | ~2-4 小時 | RTX 3060+ |
| 訓練 1 個影子模型（100 epochs） | ~1-2 小時 | RTX 3060+ |
| 訓練 64 個影子模型 | ~64-128 小時 | 需要多 GPU 或分布式 |

### Q2：我可以在 CPU 上運行嗎？

可以，但會非常慢（慢 50-100 倍）。建議使用 GPU。

### Q3：如何禁用 Weights & Biases（wandb）日誌？

在所有命令中添加 `--nowandb` 參數（已在上述命令中包含）。

### Q4：「找不到檢查點」錯誤

確保：
1. 已下載模型到 `saved_models/wrn28-10/` 目錄
2. 文件名正確（`wrn28-10_target_last.pth` 和 `wrn28-10_shadow_*.pth`）
3. 使用正確的 `--name` 參數（應為 `wrn28-10`）

### Q5：結果比論文差

- **原因 1**：樣本數不足（確保使用 `--start 0 --end 5000`）
- **原因 2**：模型不同（確保使用官方下載的模型）
- **原因 3**：超參數調整（論文使用的精確超參數在命令中給出）

---

## 攻擊原理簡介

### 三種方法對比

| 方法 | 特點 | 優勢 | 劣勢 |
|------|------|------|------|
| **LiRA** | 統計影子模型輸出 | 簡單、快速 | 精度較低 |
| **Canary Online** | 實時生成對抗樣本 | 精度最高 | 需要持續訪問模型 |
| **Canary Offline** | 預先生成對抗樣本 | 不需要實時訪問 | 精度略低於 Online |

### 為什麼有效？

1. **過擬合**：模型對訓練樣本「記住」，預測更自信
2. **集成優勢**：64 個影子模型中約 32 個含有目標樣本
3. **對抗樣本**：Canary 樣本放大成員/非成員差異

---

## 快速開始總結

```powershell
# 1. 創建並激活環境
conda create -n canary python=3.10 -y
conda activate canary

# 2. 安裝依賴
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn wandb pynvml einops timm tensorboard

# 3. 下載模型（或訓練）
# 從 Google Drive 下載並解壓到 saved_models/wrn28-10/

# 4. 運行攻擊（快速測試）
python gen_canary.py --name wrn28-10 --save_name test_offline --offline --num_shadow 64 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adam --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --stop_loss 23 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 10 --nowandb

# 5. 查看結果（輸出會顯示在終端）
```

---

## 相關文件

- [原始 README](README.md) - 論文信息和引用
- [train.py](train.py) - 影子模型訓練
- [gen_canary.py](gen_canary.py) - Canary 攻擊實現
- [models/inferencemodel.py](models/inferencemodel.py) - 模型推理包裝

---

## 引用

如果使用本代碼，請引用原論文：

```bibtex
@article{wen2022canary,
  title={Canary in a Coalmine: Better Membership Inference with Ensembled Adversarial Queries},
  author={Wen, Yuxin and Torrellas, Josep},
  journal={arXiv preprint arXiv:2210.10750},
  year={2022}
}
```

---

**最後更新**：2026 年 1 月 14 日
