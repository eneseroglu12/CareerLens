from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.database import _merge_overlapping_chunks, initialize_database, replace_source_chunks, search_chunks, search_chunks_text
from src.document_utils import _chunk_words
from src.foundry_runtime import CHAT_FALLBACK_MAX_LENGTHS, CHAT_MAX_LENGTH, CHAT_MODEL_ALIAS, EMBEDDING_MODEL_CPU_VARIANT, FoundryRuntime
from src.query_router import route_question
from src.rag_engine import LocalRAGEngine
from src.requirement_guard import (
    Requirement,
    EvidenceUnit,
    build_candidate_evidence,
    extract_cv_evidence_units,
    extract_job_requirements,
    extracted_requirements_are_safe,
    summarize_guarded_results,
    validate_classification,
)
from src.text_utils import keyword_overlap


class CoreTests(unittest.TestCase):

    def test_chat_context_is_memory_bounded(self):
        self.assertEqual(CHAT_MODEL_ALIAS, "qwen2.5-0.5b")
        self.assertEqual(CHAT_MAX_LENGTH, 2048)
        self.assertEqual(CHAT_FALLBACK_MAX_LENGTHS, (1536, 1024))
        self.assertTrue(all(length < CHAT_MAX_LENGTH for length in CHAT_FALLBACK_MAX_LENGTHS))

    def test_ui_starts_clean_without_demo_loader(self):
        app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("Türkçe demo belgelerini yükle", app_text)
        self.assertNotIn("ingest_turkish_demo_documents", app_text)
        self.assertIn("Belgelerini ekle", app_text)
        self.assertIn("Eşleşmeyi analiz et", app_text)

    def test_bad_allocation_is_recognized_as_memory_error(self):
        exc = RuntimeError("OnnxChatGenerator::CreateImpl failed to create generator: bad allocation")
        self.assertTrue(FoundryRuntime._is_memory_allocation_error(exc))

    def test_embedding_is_pinned_to_cpu_variant(self):
        self.assertEqual(
            EMBEDDING_MODEL_CPU_VARIANT,
            "qwen3-embedding-0.6b-generic-cpu:1",
        )

    def test_chunking_has_overlap(self):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = _chunk_words(text, max_words=100, overlap_words=20)
        self.assertGreater(len(chunks), 1)
        self.assertIn("word80", chunks[1])

    def test_merge_overlapping_chunks_removes_duplicate_overlap(self):
        merged = _merge_overlapping_chunks([
            "alpha beta gamma delta",
            "gamma delta epsilon zeta",
        ])
        self.assertEqual(merged, "alpha beta gamma delta epsilon zeta")

    def test_keyword_overlap(self):
        result = keyword_overlap(
            "SQL Excel requirements analysis",
            "SQL Excel Jira requirements UAT",
        )
        self.assertIn("sql", result["matched"])
        self.assertIn("jira", result["missing"])

    def test_router_cv_only(self):
        route = route_question("What technical skills are mentioned in the candidate's CV?")
        self.assertEqual(route["doc_types"], ["cv"])
        self.assertFalse(route["comparison"])

    def test_router_job_only(self):
        route = route_question("What are the main requirements of the job description?")
        self.assertEqual(route["doc_types"], ["job"])
        self.assertFalse(route["comparison"])

    def test_router_cv_job_comparison(self):
        route = route_question("What skills are missing from my CV for this role?")
        self.assertEqual(route["doc_types"], ["cv", "job"])
        self.assertTrue(route["comparison"])

    def test_router_turkish_comparison(self):
        route = route_question("CV'm bu iş ilanına ne kadar uygun?")
        self.assertEqual(route["doc_types"], ["cv", "job"])
        self.assertTrue(route["comparison"])

    def test_router_cv_improvement_is_cv_not_guide(self):
        route = route_question("CV'de neler geliştirilmeli?")
        self.assertEqual(route["doc_types"], ["cv"])
        self.assertEqual(route["label"], "Yalnız CV")
        self.assertFalse(route["comparison"])

    def test_cv_improvement_question_uses_cv_when_no_guide_exists(self):
        class FakeRuntime:
            def embed_texts(self, texts, progress_callback=None):
                raise AssertionError("Q&A query path should stay embedding-free")

            def chat(self, system_prompt, user_prompt, **kwargs):
                self.user_prompt = user_prompt
                return "CV kanıtlarına dayalı geliştirme önerisi"

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            initialize_database(db_path)
            replace_source_chunks(
                db_path,
                "candidate_cv.txt",
                "cv",
                [
                    {
                        "page": None,
                        "chunk_index": 0,
                        "content": "DENEYİM İş analizi stajı. BECERİLER SQL Excel. DİLLER İngilizce B2.",
                        "embedding": [1.0, 0.0],
                    }
                ],
            )

            engine = LocalRAGEngine()
            runtime = FakeRuntime()
            engine.runtime = runtime
            answer, results, route = engine.answer(
                "CV'de neler geliştirilmeli?",
                db_path=db_path,
                top_k=3,
                doc_types=None,
            )

            self.assertEqual(route["doc_types"], ["cv"])
            self.assertEqual(route["label"], "Yalnız CV")
            self.assertIn("geliştirme alanları", answer.lower())
            self.assertTrue(results)
            self.assertTrue(all(item["doc_type"] == "cv" for item in results))
            self.assertFalse(hasattr(runtime, "user_prompt"))
            self.assertEqual(route.get("answer_mode"), "Hızlı kanıt yanıtı")

    def test_engine_auto_route_blocks_wrong_doc_type(self):
        class FakeRuntime:
            def embed_texts(self, texts, progress_callback=None):
                raise AssertionError("Q&A should not load the embedding model at query time")

            def chat(self, system_prompt, user_prompt, **kwargs):
                self.max_output_tokens = kwargs.get("max_output_tokens")
                return "kanıta dayalı"

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            initialize_database(db_path)
            replace_source_chunks(
                db_path,
                "candidate_cv.txt",
                "cv",
                [{
                    "page": None,
                    "chunk_index": 0,
                    "content": "SKILLS SQL Python Excel",
                    "embedding": [0.80, 0.20],
                }],
            )
            replace_source_chunks(
                db_path,
                "career_guide.md",
                "guide",
                [{
                    "page": None,
                    "chunk_index": 0,
                    "content": "General skills advice",
                    "embedding": [1.0, 0.0],
                }],
            )

            engine = LocalRAGEngine()
            engine.runtime = FakeRuntime()
            answer, results, route = engine.answer(
                "What technical skills are mentioned in the candidate's CV?",
                db_path=db_path,
                top_k=4,
                doc_types=None,
            )
            self.assertIn("teknik beceriler", answer.lower())
            self.assertIn("SQL", answer)
            self.assertEqual(route["doc_types"], ["cv"])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source"], "candidate_cv.txt")
            self.assertTrue(all(item["doc_type"] == "cv" for item in results))

    def test_comparison_retrieves_both_document_sides_independently(self):
        class FakeRuntime:
            def embed_texts(self, texts, progress_callback=None):
                return [[1.0, 0.0] for _ in texts]

            def chat(self, system_prompt, user_prompt, **kwargs):
                return '[{"requirement_id":"R1","status":"matched","evidence_ids":["E1"]}]'

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            initialize_database(db_path)
            replace_source_chunks(
                db_path, "cv.txt", "cv",
                [
                    {"page": None, "chunk_index": 0, "content": "SKILLS\nSQL Excel", "embedding": [1.0, 0.0]},
                    {"page": None, "chunk_index": 1, "content": "SQL Excel analytics", "embedding": [0.99, 0.01]},
                ],
            )
            replace_source_chunks(
                db_path, "job.txt", "job",
                [{"page": None, "chunk_index": 0, "content": "Requirements\n- Working knowledge of SQL and Excel.", "embedding": [0.90, 0.10]}],
            )

            engine = LocalRAGEngine()
            engine.runtime = FakeRuntime()
            answer, results, route = engine.answer(
                "How well does my CV match this job?",
                db_path=db_path,
                top_k=2,
            )
            self.assertTrue(route["comparison"])
            self.assertEqual(route["selected_cv"], "cv.txt")
            self.assertEqual(route["selected_job"], "job.txt")
            self.assertTrue(any(item["doc_type"] == "cv" for item in results))
            self.assertTrue(any(item["doc_type"] == "job" for item in results))
            self.assertIn("Genel değerlendirme", answer)


    def test_guarded_match_does_not_require_chat_model_for_structured_comparison(self):
        class FakeRuntime:
            def embed_texts(self, texts, progress_callback=None):
                # Flat vectors deliberately force the deterministic lexical/concept
                # rescue to prove the fallback does not depend on LLM output.
                return [[1.0, 0.0] for _ in texts]

            def chat(self, *args, **kwargs):
                raise RuntimeError("OnnxChatGenerator::CreateImpl failed to create generator: bad allocation")

        root = Path(__file__).resolve().parents[1]
        cv_text = (root / "tests" / "fixtures" / "sample_cv.txt").read_text(encoding="utf-8")
        job_text = (root / "tests" / "fixtures" / "sample_job_description.txt").read_text(encoding="utf-8")
        engine = LocalRAGEngine()
        engine.runtime = FakeRuntime()
        result = engine._guarded_match_from_texts(cv_text, job_text)
        self.assertIn("Hızlı kanıt motoru", result["classifier_mode"])
        self.assertEqual(result["summary"]["fit"], "Güçlü uyum")
        self.assertEqual(result["summary"]["required_counts"]["not_evidenced"], 0)

    def test_sqlite_cosine_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            initialize_database(db_path)
            replace_source_chunks(
                db_path,
                "cv.txt",
                "cv",
                [{
                    "page": None,
                    "chunk_index": 0,
                    "content": "SQL and analytics",
                    "embedding": [1.0, 0.0],
                }],
            )
            replace_source_chunks(
                db_path,
                "guide.md",
                "guide",
                [{
                    "page": None,
                    "chunk_index": 0,
                    "content": "general career advice",
                    "embedding": [0.99, 0.01],
                }],
            )
            unrestricted = search_chunks(
                db_path,
                np.array([1.0, 0.0], dtype=np.float32),
                top_k=2,
            )
            self.assertEqual(len(unrestricted), 2)
            cv_only = search_chunks(
                db_path,
                np.array([1.0, 0.0], dtype=np.float32),
                top_k=2,
                doc_types=["cv"],
            )
            self.assertEqual(len(cv_only), 1)
            self.assertEqual(cv_only[0]["doc_type"], "cv")

    def test_fast_text_search_ranks_literal_evidence_without_query_embedding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            initialize_database(db_path)
            replace_source_chunks(
                db_path,
                "cv.txt",
                "cv",
                [
                    {"page": None, "chunk_index": 0, "content": "Python pandas ile veri analizi yaptım", "embedding": [1.0, 0.0]},
                    {"page": None, "chunk_index": 1, "content": "Kulüp etkinlikleri düzenledim", "embedding": [0.0, 1.0]},
                ],
            )
            results = search_chunks_text(
                db_path,
                "Python ile veri analizi",
                top_k=1,
                doc_types=["cv"],
            )
            self.assertEqual(results[0]["chunk_index"], 0)
            self.assertGreater(results[0]["score"], 0.0)

    def test_requirement_extractor_separates_required_and_preferred(self):
        job = """Requirements
- Working knowledge of SQL and Excel.
- Clear written and verbal English communication.
Nice to have
- Experience with Power BI.
"""
        items = extract_job_requirements(job)
        self.assertEqual([x.category for x in items], ["required", "required", "preferred"])
        self.assertIn("SQL", items[0].text)
        self.assertIn("Power BI", items[2].text)

    def test_flattened_job_description_is_split_into_real_requirements(self):
        job = (
            "SAMPLE JOB DESCRIPTION — JUNIOR BUSINESS ANALYST We are looking for a Junior Business Analyst. "
            "Responsibilities - Gather and document business requirements from stakeholders. "
            "- Map current and future-state processes. - Support UAT planning and issue tracking. "
            "Requirements - University student or recent graduate in MIS or Business Analytics. "
            "- Strong analytical and problem-solving skills. - Working knowledge of SQL and Excel. "
            "- Familiarity with process modeling and software testing. "
            "- Clear written and verbal English communication. "
            "Nice to have - Experience with Power BI. - Exposure to Agile product development and Jira."
        )
        items = extract_job_requirements(job)
        required = [x for x in items if x.category == "required"]
        preferred = [x for x in items if x.category == "preferred"]
        self.assertEqual(len(required), 5)
        self.assertEqual(len(preferred), 2)
        self.assertTrue(any("SQL and Excel" in x.text for x in required))
        self.assertFalse(any("Gather and document" in x.text for x in items))
        self.assertTrue(extracted_requirements_are_safe(items, job))

    def test_turkish_flattened_job_description_is_split(self):
        job = (
            "ÖRNEK İŞ İLANI Sorumluluklar - İş gereksinimlerini toplamak. - Süreçleri haritalamak. "
            "Gereksinimler - MIS veya İş Analitiği öğrencisi ya da yeni mezun olmak. "
            "- Güçlü analitik düşünme ve problem çözme becerileri. - SQL ve Excel bilgisi. "
            "- Süreç modelleme ve yazılım testine aşina olmak. - Yazılı ve sözlü İngilizce iletişim becerisi. "
            "Tercih edilen nitelikler - Power BI deneyimi. - Agile ve Jira aşinalığı."
        )
        items = extract_job_requirements(job)
        self.assertEqual(sum(x.category == "required" for x in items), 5)
        self.assertEqual(sum(x.category == "preferred" for x in items), 2)
        self.assertFalse(any("gereksinimlerini toplamak" in x.text.lower() for x in items))

    def test_suspicious_single_giant_requirement_is_rejected(self):
        huge = "Requirements - " + ("çok uzun tek parça gereksinim metni " * 30)
        reqs = [Requirement("R1", huge, "required")]
        self.assertFalse(extracted_requirements_are_safe(reqs, huge))

    def test_structural_chunking_preserves_headings_and_newlines(self):
        text = "BAŞLIK\nGEREKSİNİMLER\n- SQL ve Excel bilgisi.\n- İngilizce iletişim."
        chunks = _chunk_words(text, max_words=100, overlap_words=20)
        self.assertEqual(len(chunks), 1)
        self.assertIn("GEREKSİNİMLER\n- SQL", chunks[0])

    def test_cv_evidence_units_are_auditable(self):
        cv = """SKILLS
SQL, Python, Excel.
EXPERIENCE
- Used SQL to validate customer records.
"""
        units = extract_cv_evidence_units(cv)
        self.assertEqual(units[0].id, "E1")
        self.assertTrue(any("Used SQL" in unit.text for unit in units))

    def test_guard_rejects_invented_evidence_id(self):
        req = Requirement("R1", "Working knowledge of SQL and Excel.", "required")
        candidates = {
            "R1": [
                {"evidence_id": "E1", "text": "SQL, Excel", "similarity": 0.95}
            ]
        }
        raw = '[{"requirement_id":"R1","status":"matched","evidence_ids":["E999"]}]'
        guarded = validate_classification(raw, [req], candidates)
        self.assertEqual(guarded[0]["status"], "matched")
        self.assertEqual(guarded[0]["evidence"][0]["evidence_id"], "E1")

    def test_guard_downgrades_unsupported_matched_claim(self):
        req = Requirement("R1", "Experience with Power BI dashboarding.", "required")
        candidates = {
            "R1": [
                {"evidence_id": "E1", "text": "Used SQL to validate records.", "similarity": 0.42}
            ]
        }
        raw = '[{"requirement_id":"R1","status":"matched","evidence_ids":["E1"]}]'
        guarded = validate_classification(raw, [req], candidates)
        self.assertEqual(guarded[0]["status"], "not_evidenced")
        self.assertEqual(guarded[0]["evidence"], [])

    def test_overall_fit_is_computed_from_guarded_statuses(self):
        items = [
            {"category": "required", "status": "matched"},
            {"category": "required", "status": "partial"},
            {"category": "required", "status": "not_evidenced"},
        ]
        summary = summarize_guarded_results(items)
        self.assertNotEqual(summary["fit"], "Güçlü uyum")
        self.assertEqual(summary["required_counts"]["not_evidenced"], 1)

    def test_guard_corrects_false_negative_for_explicit_sql_excel(self):
        req = Requirement("R1", "SQL ve Excel konusunda çalışma bilgisine sahip olmak.", "required")
        candidates = {
            "R1": [
                {
                    "evidence_id": "E1",
                    "text": "BECERİLER SQL, Python, Excel, Power BI.",
                    "similarity": 0.74,
                    "lexical_score": 1.0,
                }
            ]
        }
        raw = '[{"requirement_id":"R1","status":"not_evidenced","evidence_ids":[]}]'
        guarded = validate_classification(raw, [req], candidates)
        self.assertEqual(guarded[0]["status"], "matched")
        self.assertEqual(guarded[0]["evidence"][0]["evidence_id"], "E1")

    def test_degree_requirement_uses_or_alternatives(self):
        req = Requirement(
            "R1",
            "MIS, İş Analitiği, Endüstri Mühendisliği veya ilgili bir alanda üniversite öğrencisi ya da yeni mezun olmak.",
            "required",
        )
        candidates = {
            "R1": [
                {"evidence_id": "E1", "text": "İş Analitiği Öğrencisi", "similarity": 0.76, "lexical_score": 0.8},
                {"evidence_id": "E2", "text": "İş Analitiği Lisans Programı, Örnek Üniversite, 2023–Devam ediyor.", "similarity": 0.74, "lexical_score": 0.7},
            ]
        }
        raw = '[{"requirement_id":"R1","status":"not_evidenced","evidence_ids":[]}]'
        guarded = validate_classification(raw, [req], candidates)
        self.assertEqual(guarded[0]["status"], "matched")

    def test_turkish_demo_expected_guarded_outcome(self):
        root = Path(__file__).resolve().parents[1]
        cv_text = (root / "tests" / "fixtures" / "sample_cv.txt").read_text(encoding="utf-8")
        job_text = (root / "tests" / "fixtures" / "sample_job_description.txt").read_text(encoding="utf-8")
        requirements = extract_job_requirements(job_text)
        evidence_units = extract_cv_evidence_units(cv_text)
        # The deterministic guard is tested against the literal CV units. The
        # runtime candidate builder supplies a semantic shortlist of these same
        # auditable rows.
        all_rows = [
            {"evidence_id": unit.id, "text": unit.text, "similarity": 0.76, "lexical_score": 0.0}
            for unit in evidence_units
        ]
        candidates = {req.id: list(all_rows) for req in requirements}
        raw = "[" + ",".join(
            f'{{"requirement_id":"{req.id}","status":"not_evidenced","evidence_ids":[]}}'
            for req in requirements
        ) + "]"
        guarded = validate_classification(raw, requirements, candidates)
        required_statuses = [x["status"] for x in guarded if x["category"] == "required"]
        preferred_statuses = [x["status"] for x in guarded if x["category"] == "preferred"]
        self.assertEqual(required_statuses, ["matched", "matched", "matched", "matched", "partial"])
        self.assertEqual(preferred_statuses, ["matched", "not_evidenced"])
        summary = summarize_guarded_results(guarded)
        self.assertEqual(summary["fit"], "Güçlü uyum")
        self.assertEqual(summary["required_counts"]["not_evidenced"], 0)

    def test_turkish_capital_i_normalization_finds_business_analytics_and_english(self):
        reqs = [
            Requirement("R1", "İş Analitiği öğrencisi olmak.", "required"),
            Requirement("R2", "Yazılı ve sözlü İngilizce iletişim becerisi.", "required"),
        ]
        candidates = {
            "R1": [{"evidence_id": "E1", "text": "İş Analitiği Öğrencisi", "similarity": 0.7, "lexical_score": 0.8}],
            "R2": [{"evidence_id": "E2", "text": "İngilizce (ileri seviye)", "similarity": 0.7, "lexical_score": 0.8}],
        }
        raw = '[{"requirement_id":"R1","status":"not_evidenced","evidence_ids":[]},{"requirement_id":"R2","status":"not_evidenced","evidence_ids":[]}]'
        guarded = validate_classification(raw, reqs, candidates)
        # The first generic requirement has related evidence; the second is
        # intentionally partial because proficiency alone does not prove both
        # written and verbal communication.
        self.assertIn(guarded[0]["status"], {"matched", "partial"})
        self.assertEqual(guarded[1]["status"], "partial")

    def test_candidate_builder_lexically_rescues_explicit_skill_line(self):
        reqs = [Requirement("R1", "Süreç modelleme ve yazılım testine aşina olmak.", "required")]
        evidence = [
            EvidenceUnit("E1", "Genel kariyer özeti."),
            EvidenceUnit("E2", "Takım çalışmasına katkı sağladı."),
            EvidenceUnit("E3", "Rapor hazırladı."),
            EvidenceUnit("E4", "Sunum yaptı."),
            EvidenceUnit("E5", "SQL sorguları kullandı."),
            EvidenceUnit("E6", "Süreç haritalama, UAT ve yazılım testi deneyimi."),
            EvidenceUnit("E7", "Operasyon ekibiyle çalıştı."),
        ]
        # Give the explicit skill line a deliberately mediocre semantic score.
        req_vec = [[1.0, 0.0]]
        ev_vecs = [
            [0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [0.96, 0.04],
            [0.95, 0.05], [0.65, 0.35], [0.94, 0.06],
        ]
        candidates = build_candidate_evidence(reqs, evidence, req_vec, ev_vecs, top_n=4)
        ids = [row["evidence_id"] for row in candidates["R1"]]
        self.assertIn("E6", ids)

    def test_turkish_demo_builder_and_guard_work_even_with_flat_semantic_scores(self):
        root = Path(__file__).resolve().parents[1]
        cv_text = (root / "tests" / "fixtures" / "sample_cv.txt").read_text(encoding="utf-8")
        job_text = (root / "tests" / "fixtures" / "sample_job_description.txt").read_text(encoding="utf-8")
        requirements = extract_job_requirements(job_text)
        evidence_units = extract_cv_evidence_units(cv_text)
        requirement_vectors = [[1.0, 0.0] for _ in requirements]
        evidence_vectors = [[0.0, 1.0] for _ in evidence_units]
        candidates = build_candidate_evidence(
            requirements, evidence_units, requirement_vectors, evidence_vectors, top_n=6
        )
        raw = "[" + ",".join(
            f'{{"requirement_id":"{req.id}","status":"not_evidenced","evidence_ids":[]}}'
            for req in requirements
        ) + "]"
        guarded = validate_classification(raw, requirements, candidates)
        required_statuses = [x["status"] for x in guarded if x["category"] == "required"]
        preferred_statuses = [x["status"] for x in guarded if x["category"] == "preferred"]
        self.assertEqual(required_statuses, ["matched", "matched", "matched", "matched", "partial"])
        self.assertEqual(preferred_statuses, ["matched", "not_evidenced"])


    def test_short_cv_language_line_is_not_mistaken_for_heading(self):
        units = extract_cv_evidence_units("Diller: İngilizce – B2 (Üst-Orta Seviye)")
        self.assertEqual(len(units), 1)
        self.assertIn("B2", units[0].text)

    def test_c1_requirement_with_b2_cv_is_partial_not_matched(self):
        req = Requirement("R1", "En az C1 düzeyinde İngilizce bilgisine sahip olmak.", "required")
        candidates = {
            "R1": [{"evidence_id": "E1", "text": "Diller: İngilizce – B2 (Üst-Orta Seviye)", "similarity": 0.0, "lexical_score": 1.0}]
        }
        guarded = validate_classification("", [req], candidates)
        self.assertEqual(guarded[0]["status"], "partial")
        self.assertEqual(guarded[0]["evidence"][0]["evidence_id"], "E1")

    def test_plain_sql_requirement_is_full_match_when_sql_is_explicit(self):
        req = Requirement("R1", "Temel düzeyde SQL bilgisine sahip olmak.", "preferred")
        candidates = {
            "R1": [{"evidence_id": "E1", "text": "Teknik Beceriler: Python, C#, SQL, Microsoft Excel", "similarity": 0.0, "lexical_score": 1.0}]
        }
        guarded = validate_classification("", [req], candidates)
        self.assertEqual(guarded[0]["status"], "matched")

    def test_software_test_process_requirement_matches_test_planning_evidence(self):
        req = Requirement("R1", "Yazılım test süreçleri hakkında temel bilgi sahibi olmak.", "preferred")
        candidates = {
            "R1": [{"evidence_id": "E1", "text": "SPEC dokümanları, test planları ve test senaryoları hazırladım.", "similarity": 0.0, "lexical_score": 1.0}]
        }
        guarded = validate_classification("", [req], candidates)
        self.assertEqual(guarded[0]["status"], "matched")

    def test_graduate_only_requirement_does_not_full_match_current_student(self):
        req = Requirement(
            "R1",
            "Yönetim Bilişim Sistemleri veya benzeri bir lisans programından mezun olmak.",
            "required",
        )
        candidates = {
            "R1": [
                {"evidence_id": "E1", "text": "Yönetim Bilişim Sistemleri 4. sınıf öğrencisiyim.", "similarity": 0.0, "lexical_score": 1.0},
                {"evidence_id": "E2", "text": "Yönetim Bilişim Sistemleri – Lisans, 2022–2027 (Beklenen)", "similarity": 0.0, "lexical_score": 1.0},
            ]
        }
        guarded = validate_classification("", [req], candidates)
        self.assertEqual(guarded[0]["status"], "partial")

    def test_kartaca_style_job_parser_does_not_append_responsibilities_heading(self):
        job = """ARANAN NİTELİKLER
- Yönetim Bilişim Sistemleri veya benzeri bir lisans programından mezun olmak.
TERCİH EDİLEN NİTELİKLER
- Üniversite döneminde yazılım geliştirme veya proje bazlı çalışmalara katılmış olmak.
GÖREV VE SORUMLULUKLAR
- İş gereksinimlerinin hazırlanmasına destek vermek.
"""
        reqs = extract_job_requirements(job)
        self.assertEqual(len(reqs), 2)
        self.assertFalse(any("GÖREV VE" in r.text for r in reqs))


    def test_documentation_plus_knowledge_sharing_is_partial_with_docs_only(self):
        req = Requirement("R1", "Dokümantasyona ve bilgi paylaşımına önem vermek.", "required")
        candidates = {
            "R1": [{"evidence_id": "E1", "text": "Fonksiyonel analiz ve SPEC dokümanları, test planları ve test senaryoları hazırladım.", "similarity": 0.0, "lexical_score": 1.0}]
        }
        guarded = validate_classification("", [req], candidates)
        self.assertEqual(guarded[0]["status"], "partial")

    def test_compound_evidence_keeps_one_row_per_concept(self):
        req = Requirement("R1", "Müşteri ihtiyaçlarını anlamaya istekli olmak ve etkili iletişim kurabilmek.", "required")
        candidates = {
            "R1": [
                {"evidence_id": "E1", "text": "İş ihtiyaçlarını analiz ettim.", "similarity": 0.0, "lexical_score": 1.0},
                {"evidence_id": "E2", "text": "İş gereksinimlerini analiz ettim.", "similarity": 0.0, "lexical_score": 1.0},
                {"evidence_id": "E3", "text": "Ekip içi iletişim ve koordinasyonda görev aldım.", "similarity": 0.0, "lexical_score": 1.0},
            ]
        }
        guarded = validate_classification("", [req], candidates)
        ids = [row["evidence_id"] for row in guarded[0]["evidence"]]
        self.assertEqual(guarded[0]["status"], "partial")
        self.assertIn("E1", ids)
        self.assertIn("E3", ids)

    def test_direct_customer_need_plus_communication_is_full_match(self):
        req = Requirement("R1", "Müşteri ihtiyaçlarını anlamak ve etkili iletişim kurmak.", "required")
        candidates = {
            "R1": [
                {"evidence_id": "E1", "text": "Müşteri ihtiyaçlarını analiz ederek gereksinimleri çıkardım.", "similarity": 0.0, "lexical_score": 1.0},
                {"evidence_id": "E2", "text": "Müşterilerle ve ekiplerle düzenli iletişim kurdum.", "similarity": 0.0, "lexical_score": 1.0},
            ]
        }
        guarded = validate_classification("", [req], candidates)
        self.assertEqual(guarded[0]["status"], "matched")

    def test_documentation_suffix_is_recognized_and_docs_only_is_partial(self):
        req = Requirement("R1", "Dokümantasyona ve bilgi paylaşımına önem vermek.", "required")
        candidates = {
            "R1": [
                {"evidence_id": "E1", "text": "Fonksiyonel analiz ve SPEC dokümanları, test planları ve test senaryoları hazırladım.", "similarity": 0.0, "lexical_score": 1.0}
            ]
        }
        guarded = validate_classification("", [req], candidates)
        self.assertEqual(guarded[0]["status"], "partial")
        self.assertEqual(guarded[0]["evidence"][0]["evidence_id"], "E1")
        self.assertIn("doküman hazırlama", guarded[0]["note"])

    def test_fast_qa_default_does_not_call_chat_model(self):
        class NoChatRuntime:
            def chat(self, *args, **kwargs):
                raise AssertionError("Fast Q&A must not load the chat model")

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            initialize_database(db_path)
            replace_source_chunks(
                db_path, "cv.txt", "cv",
                [{"page": None, "chunk_index": 0, "content": "BECERİLER SQL Excel. SQL sorguları ile test verilerini kontrol ettim. DİLLER İngilizce – C1", "embedding": [1.0, 0.0]}],
            )
            engine = LocalRAGEngine()
            engine.runtime = NoChatRuntime()
            answer, results, route = engine.answer("CV'de SQL kullanımı var mı?", db_path=db_path)
            self.assertIn("SQL için açık kanıt", answer)
            self.assertEqual(route.get("answer_mode"), "Hızlı kanıt yanıtı")
            self.assertTrue(results)

    def test_fast_qa_source_filter_uses_selected_cv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            initialize_database(db_path)
            replace_source_chunks(
                db_path, "sql_cv.txt", "cv",
                [{"page": None, "chunk_index": 0, "content": "BECERİLER SQL Excel", "embedding": [1.0, 0.0]}],
            )
            replace_source_chunks(
                db_path, "python_cv.txt", "cv",
                [{"page": None, "chunk_index": 0, "content": "BECERİLER Python pandas NumPy", "embedding": [1.0, 0.0]}],
            )
            answer, results, route = LocalRAGEngine().answer(
                "CV'deki teknik becerileri listele.",
                db_path=db_path,
                sources=["python_cv.txt"],
            )
            self.assertIn("Python", answer)
            self.assertNotIn("SQL", answer)
            self.assertEqual(route.get("selected_source"), "python_cv.txt")
            self.assertTrue(all(item["source"] == "python_cv.txt" for item in results))

    def test_fast_qa_turkish_english_question(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            initialize_database(db_path)
            replace_source_chunks(
                db_path, "cv.txt", "cv",
                [{"page": None, "chunk_index": 0, "content": "DİLLER İngilizce – B2 Türkçe – Ana Dil", "embedding": [1.0, 0.0]}],
            )
            answer, _, route = LocalRAGEngine().answer("Adayın İngilizce seviyesi nedir?", db_path=db_path)
            self.assertIn("B2", answer)
            self.assertEqual(route["doc_types"], ["cv"])


if __name__ == "__main__":
    unittest.main()
