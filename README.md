# CNN Face Detection/Explanation

This repository contains a simple Convolutional Neural Network (CNN) project for face processing, including classification, explainability, and a face engine interface.

## Project structure

- `app.py` - Main application script
- `classifier.py` - CNN classification model logic
- `face_engine.py` - Face detection/engine utilities
- `explainability.py` - Explainability methods (Grad-CAM, saliency, etc.)
- `database/` - Optional dataset or persistence storage
- `instruction.txt` / `explanation.txt` - Notes and instructions
- `requirements.txt` - Python dependencies
- `.gitignore` - Ignored files for git

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourname/CNN.git
   cd CNN
   ```
2. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
streamlit run app.py
```

If the project requires a model path or dataset input, adjust inside `app.py` or pass CLI args as implemented.

## Notes

- Keep `venv/`, `__pycache__/`, and large model/cache files out of Git; `.gitignore` already handles this.
- Update the `database/` path if you store datasets or trained model outputs there.

## Testing

Add tests under `tests/` and run via your preferred framework (e.g., `pytest`).

## Contribution

1. Fork the repository
2. Create a feature branch
3. Add tests and new code
4. Open a pull request

