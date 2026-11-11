# Mortgage Decision Engine

An educational Senior Mortgage Underwriting System for the JHU Agentic AI course.
Four specialist agents use deterministic financial tools and retrieved policy evidence; a supervisor coordinates their work, a critic reviews it, and a decision agent explains the recommendation. Required cases pause for human review.

## Local setup

Use Python 3.11 or later in VS Code. Create a virtual environment, install requirements.txt, then install this package with `python -m pip install -e .`.
Copy .env.example to .env and set the university-issued API credential locally. Never publish .env.
Use the sibling assignment files folder for the private policy PDF and supplied JSON.
Run `python -m pytest` from this repository. The tests use explicit test doubles and make no API calls.
The private submissions folder contains the executed self-contained notebook, HTML and readiness report after the build pipeline completes.

## Important interpretation

The provided expected decisions conflict with recalculated DTI. The engine retains original fixtures and reports discrepancies; it does not force expected approval labels.
See docs/decisions.md for policy precedence, assumptions, scoring and limitations.
No deployment or remote publishing is performed automatically.

## References

- LangGraph interrupt and resume: https://reference.langchain.com/python/langgraph/types/interrupt
- Chroma Python client: https://docs.trychroma.com/reference/python

## Run an interactive application

From this repository in PowerShell, after activating .venv:

```powershell
python -m mortgage_engine.cli --cases "../assignment files/mortgage_test_cases.json" --policy "../assignment files/underwriting_policies.pdf" --case 1
```

The CLI waits for your review when required. Leaving the reviewer blank keeps the application pending.
To recreate private submission artifacts, run `python scripts/build_submission.py`. This makes live API calls; notebook reviewer records are explicitly simulated for reproducibility.
The builder embeds implementation and private inputs directly in the notebook and executes a fresh kernel before exporting HTML.
