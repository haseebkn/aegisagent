"""Exercise the real dashboard against an isolated, already-built synthetic demo.

Run by scripts.demo in a child process so configuration and Streamlit caches do
not leak between tests. No narrative-provider calls or RGS decisions are made.
"""
import tempfile
from pathlib import Path
import os


def main():
    with tempfile.TemporaryDirectory() as temporary:
        # Dashboard actions use disposable cases, even when rerun manually.
        os.environ["AEGIS_CASE_DB_PATH"] = str(Path(temporary) / "cases.sqlite3")
        os.environ["AEGIS_EVIDENCE_DIR"] = str(Path(temporary) / "evidence")
        from streamlit.testing.v1 import AppTest
        from scripts.case_management import CaseStore

        app = AppTest.from_file("app.py", default_timeout=60).run()

        def healthy():
            assert not app.exception, [item.message for item in app.exception]
            assert not app.error, [item.value for item in app.error]

        def button(label):
            return next((item for item in app.button if item.label == label), None)

        healthy()
        picker = next(item for item in app.selectbox if item.label == "Choose Transaction")
        # Fraud labels help locate fixture examples; alert eligibility still
        # comes exclusively from live model inference in the dashboard.
        candidate_indexes = [
            index for index, label in enumerate(picker.options) if "FRAUD" in label
        ]
        for index in candidate_indexes:
            next(item for item in app.selectbox if item.label == "Choose Transaction").set_value(index).run()
            healthy()
            create = button("Create Investigation Case")
            if create is not None:
                create.click().run()
                break
        else:
            raise AssertionError("No real model alert available for dashboard smoke test")
        healthy()
        next(item for item in app.text_area if item.label == "Assignment rationale").set_value(
            "Scripted synthetic demonstration of the investigator assignment workflow."
        )
        button("Assign to Me & Start Review").click().run()
        healthy()
        button("Verify Case & Evidence Integrity").click().run()
        healthy()
        cases = CaseStore().list_cases()
        assert len(cases) == 1 and cases[0].status == "under_review"
        assert CaseStore().verify_integrity(cases[0].case_id)["ok"]
        print("Dashboard smoke passed: render, select, create, assign, verify; no cloud calls.")


if __name__ == "__main__":
    main()
