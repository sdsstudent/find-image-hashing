"""Module entry point: `python -m api` starts the FastAPI service.

Equivalent to `uvicorn api.main:app --host 0.0.0.0 --port 8945` but
shorter for README quickstart, demos, and ad-hoc local runs.

Honours the same env vars as the Docker CMD (PORT, HOST) so the same
launch surface works inside and outside the container.
"""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8945")),
    )


if __name__ == "__main__":
    main()
