$ErrorActionPreference = "Stop"
Write-Host "Starting QAT Training for all models..."

Write-Host "`n--- Training Micro-LM-Pico ---"
.venv\Scripts\python scripts/train_qat.py --config micro_lm_pico --epochs 50

Write-Host "`n--- Training Micro-LM-Ultra ---"
.venv\Scripts\python scripts/train_qat.py --config micro_lm_ultra --epochs 50 --resume checkpoints/Micro-LM-Ultra/Micro-LM-Ultra_qat.pth

Write-Host "`n--- Training Micro-LM-S3-Large (26M) ---"
.venv\Scripts\python scripts/train_qat.py --config micro_lm_s3_large --epochs 30

Write-Host "`nAll training complete!"
