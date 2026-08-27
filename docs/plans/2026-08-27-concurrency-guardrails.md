# Concurrency Guardrails Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent per-process connection-pool multiplication and nested logic-pool starvation.

**Architecture:** Route request decoding and synchronous analysis through a small dedicated request executor. Retain the logic executor only for `logic_run` work, and cap process and pool sizes consistently across the local startup path and environment configuration.

**Tech Stack:** Python 3, FastAPI/Uvicorn, `ThreadPoolExecutor`, unittest.

---

### Task 1: Add a regression test

**Files:**

- Create: `tests/test_concurrency_config.py`

**Step 1:** Assert that the request executor is distinct, all request dispatch calls use it, and startup/config caps are 4 and 8/8/4/4.

**Step 2:** Run `python -m unittest tests.test_concurrency_config -v` and observe failure on the old configuration.

### Task 2: Apply the bounded-concurrency configuration

**Files:**

- Modify: `apps/Cangqiong_Smart_Analyse/analyse.py`
- Modify: `.env`
- Modify: `start.sh`

**Step 1:** Create `request_executor` with `REQUEST_MAX_WORKERS=4`; leave logic work on `logic_executor`.

**Step 2:** Set Uvicorn workers to 4 and pool variables to 8/8/4/4.

### Task 3: Verify

**Files:**

- Test: `tests/test_concurrency_config.py`
- Test: `tests/test_compose.py`

**Step 1:** Re-run both tests and ensure they pass.
