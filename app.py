# Root-level entry point required by Hugging Face Spaces.
# Spaces looks for app.py at the repo root and runs it via streamlit.
# This file simply delegates to the actual app.

import runpy
runpy.run_path("app/streamlit_app.py", run_name="__main__")
