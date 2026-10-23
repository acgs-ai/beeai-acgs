# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Entries for new releases are generated with `cz bump`.

## [Unreleased]

## 0.1.0 (2026-10-23)

### Added

- `ACGSMiddleware`: a BeeAI run middleware that checks each tool call's name and validated input against an ACGS
  constitution before the tool runs, and sets the denial reason as the tool output. It follows nested runs, so
  tools of `HandoffTool` sub-agents and tools called by other tools are checked as well.
- Review handling through an injected reviewer (`approver=`, sync or async, `review_timeout=`), with
  `io_confirm_approver` for BeeAI's IO handlers; `on_review="deny"` denies without asking. No reviewer, an error
  or a timeout denies the call.
- `ACGSMiddleware.permission_requirement()`: an `AskPermissionRequirement` for `RequirementAgent` that asks only
  about calls ACGS sends to review.
- A final check right before the tool runs: if the input changed after the decision, or a denied call starts
  anyway, the run ends with `ACGSToolCallChangedError`.
- `ACGSMiddleware.evaluate` and `decide`; the `tool_call_formatter=` option.
