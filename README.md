# SimGui


Similarity Guidance for Multimodal Aspect-Based Sentiment Analysis (MABSA).

## Project Structure

- `labeling/`: scripts for data labeling and aspect/category extraction.
- `framework/`: model training, testing, preprocessing, and result collection code.

## Setup

Create and activate a Python environment, then install dependencies for the part
of the project you want to run.

For labeling:

```powershell
pip install -r labeling\requirements.txt
```

For model training and evaluation:

```powershell
pip install -r framework\requirements.txt
```

## Framework Usage

See `framework\SETUP_AND_RUN.md` for dataset layout, training commands, and
evaluation commands.
