# nlp_final

---

## File Overview

### `agents.py`
- Main script for the floor plan revision pipeline.
- Implements three core agents:
  - **Code Searcher Agent** – retrieves relevant building code sections.
  - **Code Examiner Agent** – evaluates the floor plan against the retrieved code.
  - **Lead Designer Agent** – generates an adjustment plan to revise the floor plan toward compliance.

### `qa.py`
- Contains curated QA datasets for evaluating retrieval and reasoning accuracy.
- Includes both expert-level and paraphrased amateur-level accessibility questions.

### `doc.md`
- Preprocessed Markdown file representing the structured building code (e.g., accessibility regulations).
- Used as the knowledge base for the retrieval-augmented generation system.

---

## Floor Plan Generative Model

This system integrates with the **Floor Plan Generative (FPG) model**, which generates updated layouts based on room-level adjustment plans.  
The FPG model is implemented separately and available at:  
🔗 [https://github.com/Haolan-Zhang/FineDiffPlan](https://github.com/Haolan-Zhang/FineDiffPlan)
