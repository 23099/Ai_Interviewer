"""Simple role-based authentication for the project prototype."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Account:
    email: str
    password: str
    account_type: str
    display_name: str


ACCOUNTS: dict[str, Account] = {
    "hr": Account(
        email="hr@itu.edu.pk",
        password="123",
        account_type="hr",
        display_name="HR Manager",
    ),
    "employer": Account(
        email="job_employer@itu.edu.pk",
        password="123",
        account_type="employer",
        display_name="Job Employer",
    ),
    "candidate": Account(
        email="candidate@itu.edu.pk",
        password="123",
        account_type="candidate",
        display_name="Job Candidate",
    ),
}


def authenticate(email: str, password: str, account_type: str) -> dict | None:
    account = ACCOUNTS.get(account_type)
    if not account:
        return None
    if email.strip().lower() != account.email or password != account.password:
        return None
    return {
        "email": account.email,
        "account_type": account.account_type,
        "display_name": account.display_name,
    }
