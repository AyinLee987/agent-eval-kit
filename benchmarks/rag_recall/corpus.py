"""Synthetic, general-domain corpus: a fictional company's engineering handbook.

Not medical, not real — picked specifically to avoid the sibling project's
RAG being tuned for Chinese medical text. Each document is short enough that
every ``## heading`` section becomes exactly one child chunk under
``MedicalParentChildChunker`` (see ``run_benchmark.py``'s assumption check),
which is what makes fact-substring ground truth in ``queries.py`` reliable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict


class Document(TypedDict):
    logical_id: str
    title: str
    sections: List[Tuple[str, str]]


DOCUMENTS: List[Document] = [
    {
        "logical_id": "incident-response",
        "title": "Incident Response Process",
        "sections": [
            ("Severity Levels", "A SEV1 incident must be acknowledged within 15 minutes and an incident commander assigned immediately."),
            ("Escalation Path", "If a SEV1 remains unresolved for 45 minutes, the on-call engineer must escalate to the duty manager."),
            ("Postmortems", "Every SEV1 and SEV2 incident requires a blameless postmortem published within 5 business days of resolution."),
        ],
    },
    {
        "logical_id": "on-call-rotation",
        "title": "On-Call Rotation Policy",
        "sections": [
            ("Rotation Schedule", "Primary on-call shifts run for one full week, starting Monday morning and handing off the following Monday."),
            ("Compensation", "Engineers on the primary rotation receive a flat stipend of 200 dollars per week regardless of how many pages they receive."),
            ("Handoff", "The outgoing on-call engineer must write a handoff summary of any open issues before the rotation changes hands."),
        ],
    },
    {
        "logical_id": "deployment-pipeline",
        "title": "Deployment Pipeline",
        "sections": [
            ("Canary Releases", "New builds are first rolled out to 5 percent of production traffic for 30 minutes before a full rollout begins."),
            ("Automatic Rollback", "If error rates exceed 2 percent above baseline during the canary window, the pipeline automatically rolls back the release."),
            ("Payments Approval", "Deployments to the payments service additionally require a second engineer's sign-off before promotion to production."),
        ],
    },
    {
        "logical_id": "code-review-policy",
        "title": "Code Review Policy",
        "sections": [
            ("Required Reviewers", "Every pull request needs approval from at least one reviewer outside the author's own team before it can merge."),
            ("Review SLA", "Reviewers are expected to leave their first feedback within one business day of a review being requested."),
            ("Large Changes", "Pull requests touching more than 800 lines should be split into smaller, independently reviewable commits."),
        ],
    },
    {
        "logical_id": "data-retention-policy",
        "title": "Data Retention Policy",
        "sections": [
            ("Log Retention", "Application logs are retained for 90 days before they are automatically deleted from the logging system."),
            ("Backup Retention", "Nightly database backups are kept for 35 days, and monthly snapshots are kept for 2 years."),
            ("Deletion Requests", "A verified user data deletion request must be fully processed within 30 days of being received."),
        ],
    },
    {
        "logical_id": "security-incident-classification",
        "title": "Security Incident Classification",
        "sections": [
            ("Severity Definitions", "A P0 security incident involves confirmed unauthorized access to customer data and requires immediate executive notification."),
            ("Reporting Window", "Any employee who suspects a security incident must report it to the security team within one hour of noticing it."),
            ("Communication Review", "Customer-facing communication about a confirmed breach must be reviewed by legal before it is ever sent out."),
        ],
    },
    {
        "logical_id": "expense-reimbursement",
        "title": "Expense Reimbursement Policy",
        "sections": [
            ("Meal Limits", "Meals during business travel are reimbursed up to 75 dollars per day without needing additional approval."),
            ("Approval Threshold", "Any single expense over 500 dollars requires prior written approval from a manager before it is booked."),
            ("Receipt Deadline", "Original receipts must be submitted within 14 days of the expense for reimbursement to be processed."),
        ],
    },
    {
        "logical_id": "vpn-remote-access",
        "title": "VPN and Remote Access Policy",
        "sections": [
            ("Authentication", "All VPN connections require multi-factor authentication in addition to a personal client certificate."),
            ("Device Requirements", "Only company-managed laptops with disk encryption enabled are allowed to connect to the production VPN."),
            ("Session Limits", "VPN sessions automatically time out and disconnect after 12 hours of inactivity."),
        ],
    },
    {
        "logical_id": "database-backup-schedule",
        "title": "Database Backup Schedule",
        "sections": [
            ("Backup Frequency", "Primary databases are backed up every 6 hours in addition to continuous transaction log shipping."),
            ("Restore Testing", "A full restore from backup is tested in a staging environment at least once every quarter."),
            ("Offsite Copies", "Encrypted backup copies are replicated to a secondary region within 24 hours of being created."),
        ],
    },
    {
        "logical_id": "api-rate-limits",
        "title": "API Rate Limits",
        "sections": [
            ("Standard Tier", "Standard tier API keys are limited to 60 requests per minute per key."),
            ("Enterprise Tier", "Enterprise tier customers can request a burst allowance of up to 500 requests per minute for short periods."),
            ("Throttling Response", "Requests beyond the limit receive an HTTP 429 response along with a Retry-After header."),
        ],
    },
    {
        "logical_id": "leave-policy",
        "title": "Vacation and Leave Policy",
        "sections": [
            ("PTO Accrual", "Full-time employees accrue 15 days of paid time off per year during their first two years of employment."),
            ("Carryover", "Up to 5 unused PTO days may be carried over into the next calendar year and no more."),
            ("Sick Leave", "Sick leave is tracked completely separately from PTO and is not subject to any yearly cap."),
        ],
    },
    {
        "logical_id": "vendor-security-review",
        "title": "Vendor Security Review Process",
        "sections": [
            ("Questionnaire", "Any new vendor handling customer data must complete a security questionnaire before a contract is signed."),
            ("Review Timeline", "The security team completes a standard vendor review within 10 business days of receiving the questionnaire."),
            ("High-Risk Vendors", "Vendors classified as high-risk require an annual re-assessment and a signed data processing agreement."),
        ],
    },
]


def render_markdown(document: Document) -> str:
    """Render a document's sections as ``## heading`` Markdown blocks.

    No top-level ``# title`` line — the document title is passed separately
    to ingestion, and including it in the body would create a redundant
    nesting level in each chunk's section path.
    """

    return "\n\n".join(f"## {heading}\n{body}" for heading, body in document["sections"])


def all_facts() -> Dict[str, str]:
    """Map every ``"logical_id::heading"`` to its body text, for sanity checks."""

    return {
        f"{doc['logical_id']}::{heading}": body
        for doc in DOCUMENTS
        for heading, body in doc["sections"]
    }
