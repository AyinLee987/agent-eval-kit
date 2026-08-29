"""Labeled queries for the RAG recall benchmark.

Each case names a ``fact`` substring instead of a chunk id, because chunk
ids are content hashes assigned during ingestion and can't be known ahead of
time. ``run_benchmark.py`` resolves each ``fact`` to the chunk(s) whose text
contains it after ingesting ``corpus.py``, and fails loudly if a fact
resolves to zero or more than one chunk — see ``resolve_relevant_ids``.

``style`` is not used by the metrics themselves; it lets the benchmark
report Recall@K separately for near-verbatim queries (where BM25 should do
fine) versus paraphrased queries that share few words with the source text
(where only a real semantic embedding should help) instead of only ever
reporting one blended average.
"""

from __future__ import annotations

from typing import List, TypedDict


class QueryCase(TypedDict):
    id: str
    query: str
    fact: str
    style: str  # "lexical" | "paraphrase"


CASES: List[QueryCase] = [
    {
        "id": "incident-ack-time",
        "query": "How quickly must a SEV1 incident be acknowledged?",
        "fact": "acknowledged within 15 minutes",
        "style": "lexical",
    },
    {
        "id": "incident-postmortem-writeup",
        "query": "What's the deadline for writing up a review after a critical outage is fixed?",
        "fact": "blameless postmortem published within 5 business days",
        "style": "paraphrase",
    },
    {
        "id": "oncall-shift-length",
        "query": "How long does a primary on-call shift last?",
        "fact": "Primary on-call shifts run for one full week",
        "style": "lexical",
    },
    {
        "id": "oncall-pager-pay",
        "query": "Do engineers get paid extra for carrying the pager?",
        "fact": "flat stipend of 200 dollars per week",
        "style": "paraphrase",
    },
    {
        "id": "canary-traffic-percent",
        "query": "What percentage of traffic does a canary release start with?",
        "fact": "rolled out to 5 percent of production traffic",
        "style": "lexical",
    },
    {
        "id": "billing-deploy-signoff",
        "query": "Who needs to sign off before shipping a change to billing?",
        "fact": "payments service additionally require a second engineer's sign-off",
        "style": "paraphrase",
    },
    {
        "id": "pr-reviewer-count",
        "query": "How many reviewers does a pull request need?",
        "fact": "approval from at least one reviewer outside the author's own team",
        "style": "lexical",
    },
    {
        "id": "huge-diff-guidance",
        "query": "What should I do if my diff is huge?",
        "fact": "Pull requests touching more than 800 lines should be split",
        "style": "paraphrase",
    },
    {
        "id": "log-retention-days",
        "query": "How long are application logs kept?",
        "fact": "Application logs are retained for 90 days",
        "style": "lexical",
    },
    {
        "id": "data-deletion-sla",
        "query": "If a customer asks us to erase their data, how fast do we have to comply?",
        "fact": "deletion request must be fully processed within 30 days",
        "style": "paraphrase",
    },
    {
        "id": "p0-definition",
        "query": "What counts as a P0 security incident?",
        "fact": "P0 security incident involves confirmed unauthorized access to customer data",
        "style": "lexical",
    },
    {
        "id": "security-report-window",
        "query": "How soon do I need to tell the security team if something looks off?",
        "fact": "report it to the security team within one hour",
        "style": "paraphrase",
    },
    {
        "id": "meal-limit",
        "query": "What's the daily meal reimbursement limit while traveling?",
        "fact": "reimbursed up to 75 dollars per day",
        "style": "lexical",
    },
    {
        "id": "expense-approval-need",
        "query": "Do I need my boss's okay before buying something expensive for work?",
        "fact": "expense over 500 dollars requires prior written approval",
        "style": "paraphrase",
    },
    {
        "id": "vpn-mfa",
        "query": "Does VPN access require multi-factor authentication?",
        "fact": "require multi-factor authentication in addition to a personal client certificate",
        "style": "lexical",
    },
    {
        "id": "personal-laptop-vpn",
        "query": "Can I connect to the company network from my personal laptop?",
        "fact": "Only company-managed laptops with disk encryption enabled",
        "style": "paraphrase",
    },
    {
        "id": "db-backup-frequency",
        "query": "How often are primary databases backed up?",
        "fact": "backed up every 6 hours",
        "style": "lexical",
    },
    {
        "id": "backup-verification",
        "query": "How do we make sure backups actually work if we ever need them?",
        "fact": "full restore from backup is tested in a staging environment",
        "style": "paraphrase",
    },
    {
        "id": "standard-rate-limit",
        "query": "What's the rate limit for standard API keys?",
        "fact": "limited to 60 requests per minute",
        "style": "lexical",
    },
    {
        "id": "throttled-response-code",
        "query": "What error do clients see when they call the API too fast?",
        "fact": "HTTP 429 response along with a Retry-After header",
        "style": "paraphrase",
    },
    {
        "id": "pto-accrual-rate",
        "query": "How many PTO days do new employees earn per year?",
        "fact": "accrue 15 days of paid time off per year",
        "style": "lexical",
    },
    {
        "id": "pto-carryover",
        "query": "Can I roll unused vacation days into next year?",
        "fact": "Up to 5 unused PTO days may be carried over",
        "style": "paraphrase",
    },
    {
        "id": "vendor-questionnaire",
        "query": "What does a new vendor need to complete before signing a contract?",
        "fact": "complete a security questionnaire before a contract is signed",
        "style": "lexical",
    },
    {
        "id": "risky-vendor-recheck",
        "query": "How often do risky suppliers need to be checked again?",
        "fact": "high-risk require an annual re-assessment",
        "style": "paraphrase",
    },
]
