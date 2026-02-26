This is my first attempt at building an end-to-end ML workflow beyond just training a model in a notebook.

The goal was to understand how training, serving, testing and containerisation fit together in a minimal but structured way.

---

## What This Project Does

- Trains a scikit-learn pipeline on the breast cancer dataset
- Saves versioned model artifacts
- Serves predictions through a FastAPI API
- Validates input shape
- Includes basic pytest tests
- Runs inside a Docker container