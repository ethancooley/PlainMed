"""Launch the PlainMed inference app.

Run locally:   python main.py
On Hugging Face Spaces the Dockerfile calls uvicorn directly on port 7860.
"""

import uvicorn


def main() -> None:
    """Start the FastAPI server."""
    uvicorn.run("app.server:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
