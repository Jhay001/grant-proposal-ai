# Project Context

## Project Name

AI Powered Grant Proposal Evaluation System

---

## Purpose

Build an AI assisted decision support system that helps donor organizations evaluate grant proposals.

The system should assist reviewers, not replace them.

The system will compare submitted proposals against donor requirements and generate structured evaluation reports.

---

# Core Workflow

## Donor Knowledge Base

Funding Guidelines.pdf
Eligibility Requirements.pdf
Evaluation Rubrics.pdf
Strategic Priorities.pdf
Past Successful Projects.pdf

↓

PDF Extraction

↓

Text Cleaning

↓

Chunking

↓

Embeddings

↓

Vector Database

---

## Proposal Evaluation

Proposal PDF

↓

Text Extraction

↓

Text Cleaning

↓

Vector Search

↓

Retrieve Relevant Donor Context

↓

Gemini Evaluation

↓

Structured JSON Output

↓

Report Generation

↓

Human Reviewer

---

# Evaluation Criteria

The system evaluates proposals using:

1. Relevance
2. Feasibility
3. Innovation
4. Sustainability
5. Budget Justification
6. Organizational Capacity
7. Expected Impact

---

# Expected Output Schema

```json
{
  "proposal_summary": "",
  "criteria_scores": {
    "relevance": 0,
    "feasibility": 0,
    "innovation": 0,
    "sustainability": 0,
    "budget_justification": 0,
    "organizational_capacity": 0,
    "expected_impact": 0
  },
  "strengths": [],
  "weaknesses": [],
  "risk_flags": [],
  "funding_alignment_analysis": "",
  "final_recommendation": ""
}
```

# Planned Tech Stack

Backend:

* Python

LLM:

* Google Gemini API

Embeddings:

* sentence-transformers

Vector Store:

* FAISS

Frontend:

* Gradio

Reporting:

* ReportLab

Deployment:

* Hugging Face Spaces

# Development Principles

1. Use modular architecture.

2. Keep functions small and focused.

3. Separate donor pipeline from proposal pipeline.

4. Store intermediate outputs for debugging.

5. Use logging wherever possible.

6. Return structured JSON from Gemini.

7. Never parse unstructured LLM responses when structured output is possible.

8. Design components to be independently testable.

# Folder Structure

src/
├── donor_pipeline/
├── proposal_pipeline/
├── retrieval/
├── evaluation/
├── reporting/
└── ui/

Each folder should contain focused modules with single responsibilities.