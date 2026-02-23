# MKJ at SemEval-2026 Task 9: Polarization Detection

This repository contains the official codebase for the paper:

**"MKJ at SemEval-2026 Task 9: A Comparative Study of Generalist, Specialist, and Ensemble Strategies for Multilingual Polarization."**

Our system implements an **Adaptive Modeling Framework** for multilingual polarization detection across 22 languages.  
The framework dynamically switches between:

- High-capacity multilingual generalists (e.g., mDeBERTa-v3)
- Monolingual specialist models
- Hybrid soft-voting ensembles  

The selection strategy is driven by linguistic and empirical performance considerations.

---

## Repository Structure

```
final_models/     # Executable Python scripts for final system configurations (per language)
predictions/      # Official test predictions generated for the shared task
requirements.txt  # Python dependencies
```

---

## Dataset & Execution

Due to licensing and shared task rules, the raw datasets are **not included** in this repository.

To reproduce experiments:

1. Download the official SemEval-2026 Task 9 dataset from the task organizers.
2. Provide the local dataset path using the `--data_dir` argument.

Example:

```bash
cd final_models
python run_eng_ensemble.py --data_dir "path/to/your/downloaded/data"
```

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended.

---

## Citation

If you use this code, prediction files, or the Adaptive Modeling Framework in your research, please cite:

```bibtex
@inproceedings{kianimoghadam2026mkj,
  title={MKJ at SemEval-2026 Task 9: A Comparative Study of Generalist, Specialist, and Ensemble Strategies for Multilingual Polarization},
  author={Kianimoghadam Jouneghani, Maziar},
  booktitle={Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```

---

## License

This repository is released for research purposes.  
Please ensure compliance with the SemEval-2026 Task 9 data usage policy.
