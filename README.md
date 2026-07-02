# MKJ at SemEval-2026 Task 9: Polarization Detection

This repository contains the codebase and final prediction files for the paper:

**"MKJ at SemEval-2026 Task 9: A Comparative Study of Generalist, Specialist, and Ensemble Strategies for Multilingual Polarization."**

**[Read the paper on arXiv](https://arxiv.org/abs/2604.21370)**

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

## Limitations and Ethical Considerations
 
- **Privacy & ToS:** We release code and **numerical predictions only** (and identifiers where applicable). We do **not** distribute raw tweet text or personal data.

---

## Citation

If you use this code, prediction files, or the language-adaptive framework in your research, please cite:

```bibtex
@inproceedings{kianimoghadam-jouneghani-2026-mkj,
    title = "{MKJ} at {S}em{E}val-2026 Task 9: A Comparative Study of Generalist, Specialist, and Ensemble Strategies for Multilingual Polarization",
    author = "Kianimoghadam Jouneghani, Maziar",
    booktitle = "Proceedings of the 20th {I}nternational {W}orkshop on {S}emantic {E}valuation (2026)",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.semeval-1.181/",
    doi = "10.18653/v1/2026.semeval-1.181",
    pages = "1398--1406",
    ISBN = "979-8-89176-414-9",
}
```

---

## License

This repository is released for research purposes.  
Please ensure compliance with the SemEval-2026 Task 9 data usage policy.
