# AI-Powered Grant Proposal Evaluation System

An intelligent decision-support system designed to assist donor organizations in the systematic evaluation of grant proposals. By leveraging Large Language Models (Gemini) and Retrieval-Augmented Generation (RAG), this tool bridges the gap between complex funding guidelines and incoming proposals, ensuring consistent, evidence-based assessments.

## Key Features

*   **Intelligent Knowledge Integration:** Embeds donor-specific guidelines, rubrics, and strategic priorities into a searchable vector database for context-aware evaluations.
*   **Automated Multi-Criteria Analysis:** Evaluates proposals across seven key pillars—Relevance, Feasibility, Innovation, Sustainability, Budget Justification, Organizational Capacity, and Impact.
*   **Structured Reporting:** Produces standardized, machine-readable JSON outputs and professional PDF evaluation reports to streamline the human review process.
*   **Modular Architecture:** Designed for scalability, separating the data ingestion pipeline from the evaluation engine.
*   **Human-in-the-Loop Design:** Acts as a powerful assistant to human reviewers, providing high-quality insights rather than making final decisions.

## Tech Stack

*   **Language:** Python
*   **AI/LLM:** Google Gemini API
*   **Embeddings & Retrieval:** `sentence-transformers`, FAISS
*   **User Interface:** Gradio
*   **Reporting:** ReportLab
*   **Deployment:** Designed for Hugging Face Spaces

## Architecture

The system is built on a robust, modular foundation:

1.  **Donor Pipeline:** Ingests and vectorizes institutional knowledge (Guidelines, Rubrics, etc.).
2.  **Proposal Pipeline:** Processes incoming submissions using text extraction and cleaning.
3.  **Retrieval Engine:** Fetches the most relevant donor context for specific proposals.
4.  **Evaluation Engine:** Gemini-powered analysis producing consistent, structured outputs.
5.  **Reporting:** Automated generation of formal PDF evaluations.

---

*Built to enhance transparency, consistency, and efficiency in the grant-making process.*
