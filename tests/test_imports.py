# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
import beeai_acgs


def test_public_api() -> None:
    assert sorted(beeai_acgs.__all__) == sorted(
        [
            "ACGSMiddleware",
            "ACGSToolCallChangedError",
            "Approver",
            "GovernanceDecision",
            "ReviewRequest",
            "ToolCallFormatter",
            "Verdict",
            "format_tool_call",
            "io_confirm_approver",
        ]
    )
    for name in beeai_acgs.__all__:
        assert getattr(beeai_acgs, name) is not None
