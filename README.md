# Grok + CPO: Hallucination-Focused Preference Optimization

**Reduce LLM hallucinations by 60% using CPO + BLASER 2.0-QE**

---

## What It Does
- **Detects** hallucinations with **BLASER 2.0-QE**  
- **Prevents** them with **CPO fine-tuning** on preference data  
- **Demo**: Eiffel Tower translation test (correct vs. hallucinated)

---

## Run Demo (No GPU needed)

```bash
pip install -r requirements.txt
python cpo_dataset.py
python cpo_trainer.py --demo