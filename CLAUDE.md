# AI Prompt Firewall — Claude Code project memory

## What this is
Final-year BSc Cybersecurity thesis, Bayero University Kano (reg. CST/22/CBS/00753). The thesis committee redirected this from a research-only proposal to require a working built system. This repo is that system: a two-stage document classifier that screens documents for indirect prompt-injection payloads before they enter a RAG pipeline's context window.

Chapter 1 of the thesis (already submitted) already describes this exact system. The code must stay consistent with what Chapter 1 claims it does. Read PROJECT_REFERENCE.docx in this repo before starting any phase — it has the full picture this file doesn't.

## Locked architecture — not up for redesign
- FastAPI middleware service, sitting between document ingestion and the LLM's context window.
- Stage 1: lightweight heuristic pre-filter — regex, pattern matching, structural scoring. Can fast-reject.
- Stage 2: CPU-efficient transformer classifier (Hugging Face Transformers), fine-tuned on Google Colab, deployed CPU-only.
- Both stages wired behind a single API endpoint.
- Benchmarked against Rebuff on precision, recall, false-positive rate, and latency.

## Hard constraints
- CPU-only at inference/runtime. No GPU, no runtime cloud-inference dependency.
- Input documents capped at roughly 350 words (encoder-only token ceiling).
- English-language documents only.
- Fine-tuning happens off-repo on Colab; only the resulting model artifact gets deployed.

## Explicitly out of scope — flag it, don't build it
- Dual-LLM verification, or any second "judge" model.
- Agentic orchestration, sub-agent routing, tool-invocation-hijack defenses.
- Memory or identity-file poisoning defenses (SOUL.md/MEMORY.md-style attacks).
- Multimodal (image/audio) injection handling.
- RAG "jamming" / blocker-document denial-of-service defenses — different threat class from injection, not this project.
- Direct chat-turn injection — relevant only as a baseline comparison point, not a build target.

This project has drifted toward some of these before (a "Cognitive Firewall" framing, a keyword-only classifier suggestion, a full mock chat-interface demo) and been pulled back each time. Treat that as a live risk, not solved history. If a request seems to call for any of the above, say so instead of building it.

## Where things stand
No code exists yet. Start at Phase 1. Full 7-phase roadmap and current target phase are in PROJECT_REFERENCE.docx.

## How Deen works
Final-year student, juggling this against exams — values small, checkable increments over one large autonomous build. Before writing code for a new phase: restate the plan in a few bullets and wait for a go-ahead. After finishing a phase: stop, summarize what was built and tested, and wait before starting the next one. Reason out loud before making an architectural call — flag weak logic or trade-offs rather than quietly picking one. Code should be clean and commented; it may end up quoted or described in the thesis's implementation chapter.
