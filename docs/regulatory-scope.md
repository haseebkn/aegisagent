# Regulatory scope and human decision boundary

## What this project does

AegisAgent predicts the fraud label in a public synthetic card-transaction dataset.
Transactions above a configurable model threshold become **investigation alerts**.
For an alert, an optional Bedrock step can organize the supplied facts and model
signals into a grounded 5W+H narrative draft.

The draft is decision support for a human investigator. It is not a legal conclusion,
a complete Suspicious Transaction Report (STR), or evidence that a report was filed.

## What the model does not decide

Under section 7 of Canada's Proceeds of Crime (Money Laundering) and Terrorist
Financing Act, the reporting duty turns on the reporting entity having reasonable
grounds to suspect that a transaction is related to money laundering, terrorist
financing, or sanctions evasion. FINTRAC guidance describes an assessment of facts,
context, and ML/TF indicators. A synthetic fraud probability is only one analytical
signal and cannot make that determination on its own.

The intended decision boundary is:

```text
model score -> alert -> authorized human review -> RGS not reached / RGS reached
                                                    |
                                                    v
                                  approved reporting workflow (out of scope)
```

The current prototype implements the first two steps and an optional drafting aid.
It does not implement case assignment, investigator disposition, RGS approval,
Schedule 1 validation, authorized submission, confirmation receipt, amendment, or
regulatory record management.

## Terminology used in this repository

- **Alert:** a transaction above the demo model threshold.
- **Narrative draft:** unapproved text generated from an alert for human review.
- **RGS determination:** a decision belonging to the reporting entity's authorized
  human process; never a model output.
- **STR:** a report prepared and submitted through an authorized FINTRAC reporting
  workflow. AegisAgent does not create or submit one.
- **Archive attempt:** the current code can attempt to copy a draft to S3. It does not
  return a durable receipt and currently fails open; it must not be described as a
  verified compliance archive.

## Data and privacy boundary

PANs and names are masked before prompting, but the current prompt still includes
gender, occupation, city, region, postal code, and exact customer and merchant
coordinates. This is not de-identification. Any real implementation would require
purpose limitation, data minimization, access controls, retention rules, audit logs,
and a privacy/security review before external model use.

## Primary references

- [PCMLTFA, section 7](https://laws-lois.justice.gc.ca/eng/acts/P-24.501/section-7.html)
- [FINTRAC suspicious transaction reporting guidance](https://fintrac-canafe.canada.ca/guidance-directives/transaction-operation/str-dod/str-dod-eng)
- [FINTRAC reporting API information](https://fintrac-canafe.canada.ca/reporting-declaration/info/api/api-eng)
- [FINTRAC record-keeping guidance](https://fintrac-canafe.canada.ca/guidance-directives/recordkeeping-document/record/fin-eng)
- [PIPEDA Principle 7 — Safeguards](https://www.priv.gc.ca/en/privacy-topics/privacy-laws-in-canada/the-personal-information-protection-and-electronic-documents-act-pipeda/p_principle/principles/p_safeguards/)

This document describes project boundaries, not legal advice or a compliance opinion.
