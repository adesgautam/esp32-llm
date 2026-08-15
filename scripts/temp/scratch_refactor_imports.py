import os
for r, d, files in os.walk("."):
    if "checkpoints" in r or ".venv" in r or "brain" in r or ".git" in r: continue
    for f in files:
        if f.endswith((".py", ".md")):
            p = os.path.join(r, f)
            with open(p, "r", encoding="utf-8") as file:
                txt = file.read()
            if "esp32_llm" in txt or "ESP32LLM" in txt:
                txt = txt.replace("from micro_lm", "from micro_lm")
                txt = txt.replace("import micro_lm", "import micro_lm")
                with open(p, "w", encoding="utf-8") as file:
                    file.write(txt)
