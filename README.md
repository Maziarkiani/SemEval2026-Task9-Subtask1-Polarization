MKJ at SemEval-2026 Task 9: Polarization Detection

This repository contains the official codebase for the paper:

"MKJ at SemEval-2026 Task 9: A Comparative Study of Generalist, Specialist, and Ensemble Strategies for Multilingual Polarization."

Our system implements an Adaptive Modeling Framework for multilingual polarization detection across 22 languages.
The framework dynamically switches between:

High-capacity multilingual generalists (e.g., mDeBERTa-v3)
Monolingual specialist models
Hybrid soft-voting ensembles
The selection strategy is driven by linguistic and empirical performance considerations.

final_models/     # Executable Python scripts for final system configurations (per language)
predictions/      # Official test predictions generated for the shared task
requirements.txt  # Python dependencies
