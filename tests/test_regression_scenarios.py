from __future__ import annotations

from pathlib import Path
import unittest

from src.rag_engine import LocalRAGEngine
from src.ui_evidence import select_evidence_snippets


ROOT = Path(__file__).resolve().parent / "fixtures" / "regression"


class RegressionScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LocalRAGEngine()

    def _run(self, cv_name: str, job_name: str):
        cv = (ROOT / "cvler" / cv_name).read_text(encoding="utf-8")
        job = (ROOT / "is_ilanlari" / job_name).read_text(encoding="utf-8")
        return self.engine._guarded_match_from_texts(cv, job)

    def _status_map(self, result):
        return {item["requirement"]: item["status"] for item in result["guarded_items"]}

    def test_business_analyst_scenario(self):
        result = self._run("cv_01_business_analyst_ayse_demir.txt", "ilan_01_business_analyst_intern.txt")
        self.assertEqual(result["summary"]["fit"], "Güçlü uyum")
        self.assertEqual(result["summary"]["required_counts"], {"matched": 8, "partial": 0, "not_evidenced": 0, "total": 8})
        self.assertEqual(result["summary"]["preferred_counts"], {"matched": 3, "partial": 0, "not_evidenced": 0, "total": 3})

    def test_data_analyst_scenario(self):
        result = self._run("cv_02_data_analyst_mehmet_kaya.txt", "ilan_02_data_analyst_intern.txt")
        self.assertEqual(result["summary"]["fit"], "Güçlü uyum")
        self.assertEqual(result["summary"]["required_counts"], {"matched": 6, "partial": 1, "not_evidenced": 0, "total": 7})
        self.assertEqual(result["summary"]["preferred_counts"], {"matched": 2, "partial": 1, "not_evidenced": 0, "total": 3})

    def test_backend_scenario(self):
        result = self._run("cv_03_software_developer_selim_arslan.txt", "ilan_03_backend_developer_intern.txt")
        self.assertEqual(result["summary"]["required_counts"], {"matched": 7, "partial": 0, "not_evidenced": 1, "total": 8})
        self.assertEqual(result["summary"]["preferred_counts"], {"matched": 3, "partial": 0, "not_evidenced": 1, "total": 4})
        statuses = self._status_map(result)
        self.assertEqual(statuses["Nesne yönelimli programlama prensiplerini bilmek."], "not_evidenced")
        self.assertEqual(statuses["Docker hakkında temel bilgi."], "not_evidenced")

    def test_operations_scenario(self):
        result = self._run("cv_04_business_generalist_zeynep_aksoy.txt", "ilan_04_project_operations_intern.txt")
        self.assertEqual(result["summary"]["fit"], "Güçlü uyum")
        self.assertEqual(result["summary"]["required_counts"], {"matched": 7, "partial": 0, "not_evidenced": 0, "total": 7})
        self.assertEqual(result["summary"]["preferred_counts"], {"matched": 1, "partial": 1, "not_evidenced": 1, "total": 3})

    def test_wrong_role_does_not_inflate_fit(self):
        result = self._run("cv_04_business_generalist_zeynep_aksoy.txt", "ilan_03_backend_developer_intern.txt")
        self.assertEqual(result["summary"]["fit"], "Sınırlı uyum")
        self.assertGreaterEqual(result["summary"]["required_counts"]["not_evidenced"], 5)

    def test_english_evidence_has_no_unrelated_skill_line(self):
        result = self._run("cv_02_data_analyst_mehmet_kaya.txt", "ilan_02_data_analyst_intern.txt")
        item = next(x for x in result["guarded_items"] if "İngilizce dokümanları" in x["requirement"])
        snippets = select_evidence_snippets(item["requirement"], item["evidence"])
        self.assertEqual(len(snippets), 1)
        self.assertIn("İngilizce", snippets[0][1])
        self.assertNotIn("SQL", snippets[0][1])
        self.assertNotIn("Power BI", snippets[0][1])

    def test_backend_postgresql_evidence_has_no_english_line(self):
        result = self._run("cv_03_software_developer_selim_arslan.txt", "ilan_03_backend_developer_intern.txt")
        item = next(x for x in result["guarded_items"] if "PostgreSQL veya SQL Server" in x["requirement"])
        snippets = select_evidence_snippets(item["requirement"], item["evidence"])
        joined = " ".join(text for _, text in snippets)
        self.assertIn("PostgreSQL", joined)
        self.assertNotIn("İngilizce", joined)


if __name__ == "__main__":
    unittest.main()
