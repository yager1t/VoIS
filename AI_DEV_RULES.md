# 🧠 AI Development Rules — Versioned, Backups & AI-Safe Workflow

## 🎯 Purpose
This document defines strict rules for developing software using AI assistants, ensuring:
- version control
- safe AI modifications
- backups and rollback
- reproducibility
- prevention of destructive changes

---

## 📌 1. Core Principle

Any change in the system must be:
- reversible
- versioned
- traceable
- tested
- documented

---

## 🔁 2. Versioning Rules

### 2.1 Semantic Versioning
MAJOR.MINOR.PATCH

- MAJOR → breaking changes
- MINOR → new features
- PATCH → bug fixes

---

### 2.2 AI Change ID

Every AI-generated change must include:

ai-change-id:
YYYYMMDD-model-hash

Example:
20260622-codex-8f3a21

---

### 2.3 Commit Rules

Format:

[feat/fix/refactor] short description

AI: yes/no  
Model: codex / gpt / local-llm  
Change-ID: xxx  

---

## 💾 3. Backup Rules

### 3.1 Mandatory backup BEFORE AI changes

Before applying any AI modification:

1. Create git commit
2. Create snapshot archive
3. Save pre-change state

---

### 3.2 Backup structure

/backups/YYYY-MM-DD/
    pre_change.zip
    post_change.zip

---

## 🧠 4. AI Execution Modes

### 4.1 EXPLAIN MODE
No code changes allowed

### 4.2 PROPOSE MODE
Generates diff only

### 4.3 APPLY MODE
Applies changes only after approval

---

## ⚠️ 5. AI Restrictions

AI must NOT:
- delete files without confirmation
- rewrite architecture without explanation
- change APIs without version bump
- overwrite modules completely

---

## 🧪 6. Validation Pipeline

Every change must pass:

1. Lint
2. Type check
3. Unit tests
4. Smoke test

---

## 🔄 7. Rollback System

If failure occurs:

- git revert preferred
- restore from /backups/ if needed

Rollback must take < 2 minutes

---

## 🧠 8. AI Memory Rules

AI must always consider:
- existing architecture
- previous commits
- dictionary of project terms
- constraints (latency, structure, APIs)

---

## 📁 9. Project Structure

/src
/tests
/backups
/docs
/ai-changes

---

## 🧩 10. AI Change Log

Every AI modification must create:

/ai-changes/YYYY-MM-DD-change-N.md

Containing:
- reason
- diff
- risk
- rollback plan

---

## 🚀 11. Stability Rule

System must always be recoverable to last stable state in < 2 minutes.

---

## 🏁 END
