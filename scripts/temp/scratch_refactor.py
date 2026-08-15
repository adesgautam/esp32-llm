import os
for r, d, files in os.walk("."):
    if "checkpoints" in r or ".venv" in r or "brain" in r or ".git" in r: continue
    for f in files:
        if f.endswith((".py", ".md")):
            p = os.path.join(r, f)
            with open(p, "r", encoding="utf-8") as file:
                txt = file.read()
            txt = txt.replace("micro_lm_pro", "micro_lm_pro").replace("micro_lm_ultra", "micro_lm_ultra").replace("micro_lm_pico", "micro_lm_pico").replace("micro_lm_max_ctx", "micro_lm_max_ctx").replace("micro_lm_s3_large", "micro_lm_s3_large").replace("micro_lm_colossus", "micro_lm_colossus")
            with open(p, "w", encoding="utf-8") as file:
                file.write(txt)
