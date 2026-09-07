from __future__ import annotations

from pathlib import Path

from .database import get_source_text, search_chunks_text
from .foundry_runtime import FoundryRuntime
from .fast_qa import build_fast_answer
from .prompts import RAG_SYSTEM_PROMPT, REQUIREMENT_CLASSIFIER_SYSTEM_PROMPT
from .query_router import route_question
from .requirement_guard import (
    build_candidate_evidence,
    build_candidate_evidence_fast,
    classification_prompt,
    extract_cv_evidence_units,
    extract_job_requirements,
    extracted_requirements_are_safe,
    render_guarded_markdown,
    summarize_guarded_results,
    validate_classification,
)
from .text_utils import keyword_overlap


class LocalRAGEngine:
    def __init__(self) -> None:
        self.runtime = FoundryRuntime()

    def embed_texts(self, texts: list[str], progress_callback=None) -> list[list[float]]:
        return self.runtime.embed_texts(texts, progress_callback=progress_callback)

    def _guarded_match_from_texts(self, cv_text: str, job_text: str) -> dict:
        """Run requirement-by-requirement CV/job comparison with evidence guards.

        Pipeline:
        1) deterministically extract explicit requirements from the job ad,
        2) split the CV into auditable evidence units,
        3) use local embeddings to shortlist CV evidence per requirement,
        4) validate each literal/concept match deterministically in Python,
        5) compute the overall fit in Python rather than trusting a free-form claim.
        """
        requirements = extract_job_requirements(job_text)
        evidence_units = extract_cv_evidence_units(cv_text)
        overlap = keyword_overlap(cv_text, job_text)

        if not requirements or not extracted_requirements_are_safe(requirements, job_text):
            return {
                "analysis": (
                    "## Genel değerlendirme: Belirsiz\n"
                    "İş ilanındaki gereksinimler güvenilir biçimde ayrı maddelere ayrıştırılamadı. "
                    "Sistem tek bir uzun paragrafı gereksinim sayıp yanlış uyum skoru üretmek yerine analizi durdurdu."
                ),
                "overlap": overlap,
                "requirements": [],
                "guarded_items": [],
                "summary": {
                    "fit": "Belirsiz",
                    "fit_note": "Açık gereksinim bulunamadı.",
                    "required_counts": {"matched": 0, "partial": 0, "not_evidenced": 0, "total": 0},
                    "preferred_counts": {"matched": 0, "partial": 0, "not_evidenced": 0, "total": 0},
                },
            }

        if not evidence_units:
            guarded_items = [
                {
                    "requirement_id": req.id,
                    "requirement": req.text,
                    "category": req.category,
                    "status": "not_evidenced",
                    "evidence": [],
                }
                for req in requirements
            ]
            summary = summarize_guarded_results(guarded_items)
            return {
                "analysis": render_guarded_markdown(guarded_items, summary),
                "overlap": overlap,
                "requirements": requirements,
                "guarded_items": guarded_items,
                "summary": summary,
            }

        # CV↔ilan matching uses a structured evidence check. Candidate evidence is
        # shortlisted and validated in Python, while Foundry Local is used for
        # document embeddings and optional local-model answers.
        candidates = build_candidate_evidence_fast(
            requirements=requirements,
            evidence_units=evidence_units,
            top_n=12,
        )

        classifier_output = ""
        classifier_mode = "Hızlı kanıt motoru + Python doğrulaması"
        classifier_warning = None
        guarded_items = validate_classification("", requirements, candidates)

        summary = summarize_guarded_results(guarded_items)
        analysis = render_guarded_markdown(guarded_items, summary)

        return {
            "analysis": analysis,
            "overlap": overlap,
            "requirements": requirements,
            "guarded_items": guarded_items,
            "summary": summary,
            "classifier_output": classifier_output,
            "classifier_mode": classifier_mode,
            "classifier_warning": classifier_warning,
        }

    @staticmethod
    def _expanded_retrieval_query(question: str) -> str:
        """Add tiny intent hints for broad questions without invoking a model."""
        q = question.lower()
        hints: list[str] = []
        if "geliştir" in q or "iyileştir" in q or "improve" in q:
            hints.extend(["deneyim", "eğitim", "beceriler", "projeler", "diller", "sorumluluk"] )
        if any(x in q for x in ("teknik beceri", "teknoloj", "skills", "araç")):
            hints.extend(["beceriler", "teknolojiler", "sql", "python", "excel", "power bi", "jira"] )
        if "ingiliz" in q or "english" in q:
            hints.extend(["ingilizce", "diller", "english"] )
        if any(x in q for x in ("gereksinim", "aranan nitelik", "requirements", "zorunlu")):
            hints.extend(["gereksinimler", "aranan nitelikler", "requirements", "nitelikler"] )
        if any(x in q for x in ("deneyim", "experience")):
            hints.extend(["deneyim", "staj", "sorumluluk", "experience"] )
        return question + (" " + " ".join(dict.fromkeys(hints)) if hints else "")

    def answer(
        self,
        question: str,
        db_path: Path,
        top_k: int = 4,
        doc_types: list[str] | None = None,
        sources: list[str] | None = None,
        use_model: bool = False,
    ) -> tuple[str, list[dict], dict]:
        """Answer with retrieval constrained to the relevant document types.

        doc_types meanings:
        - None: automatic intent routing (recommended)
        - []: explicitly search all document types
        - ["cv"], ["job"], ...: manual scope
        """
        if doc_types is None:
            route = route_question(question)
            effective_doc_types = route["doc_types"] or None
        else:
            effective_doc_types = doc_types or None
            label_map = {
                ("cv",): "Yalnız CV",
                ("job",): "Yalnız İş İlanı",
                ("guide",): "Yalnız Kariyer Rehberi",
                ("cv", "job"): "CV + İş İlanı",
                ("cv", "guide"): "CV + Kariyer Rehberi",
            }
            key = tuple(doc_types)
            automatic_comparison = set(doc_types) == {"cv", "job"} and route_question(question).get("comparison", False)
            route = {
                "doc_types": list(doc_types),
                "label": label_map.get(key, "Tüm belgeler" if not doc_types else "Manuel kapsam"),
                "reason": "Arama kapsamı kullanıcı tarafından seçildi.",
                "comparison": automatic_comparison,
            }

        retrieval_query = self._expanded_retrieval_query(question)

        # Interactive Q&A intentionally avoids loading the 0.6B embedding model
        # for every question. Documents are embedded at ingestion, while the
        # question path uses a fast local text ranker and a compact local chat
        # model. Comparison questions still retrieve both sides independently.
        if route.get("comparison"):
            side_k = max(2, (int(top_k) + 1) // 2)
            cv_results = search_chunks_text(
                db_path=db_path,
                query_text=retrieval_query,
                top_k=side_k,
                doc_types=["cv"],
                sources=sources,
            )
            job_results = search_chunks_text(
                db_path=db_path,
                query_text=retrieval_query,
                top_k=side_k,
                doc_types=["job"],
                sources=sources,
            )
            if cv_results and job_results:
                best_cv = cv_results[0]
                best_job = job_results[0]
                guarded = self._guarded_match_from_texts(
                    get_source_text(db_path, best_cv["source"]),
                    get_source_text(db_path, best_job["source"]),
                )
                route = dict(route)
                route["selected_cv"] = best_cv["source"]
                route["selected_job"] = best_job["source"]
                extras = [
                    item for item in (cv_results[1:] + job_results[1:])
                    if item["source"] in {best_cv["source"], best_job["source"]}
                ]
                selected_results = [best_cv, best_job] + extras
                selected_results = selected_results[: max(int(top_k), 2)]
                return guarded["analysis"], selected_results, route
            return (
                "Karşılaştırma için hem bir CV hem de bir iş ilanı kanıtı gerekli. Seçilen kapsamda iki belge türünden biri bulunamadı.",
                cv_results + job_results,
                route,
            )

        results = search_chunks_text(
            db_path=db_path,
            query_text=retrieval_query,
            top_k=top_k,
            doc_types=effective_doc_types,
            sources=sources,
        )
        if not results:
            return (
                "Seçilen belge kapsamında bu soruyu yanıtlamak için yeterli kanıt bulamadım.",
                [],
                route,
            )

        if not use_model:
            answer, selected_source = build_fast_answer(
                question=question,
                db_path=db_path,
                results=results,
                route=route,
            )
            route = dict(route)
            route["answer_mode"] = "Hızlı kanıt yanıtı"
            if selected_source:
                route["selected_source"] = selected_source
            return answer, results, route

        context_blocks = []
        for index, item in enumerate(results, start=1):
            page = f", sayfa {item['page']}" if item.get("page") else ""
            context_blocks.append(
                "\n".join(
                    [
                        f"=== KAYNAK [{index}] BAŞLANGIÇ ===",
                        f"Dosya: {item['source']}",
                        f"Belge türü: {item['doc_type']}{page}",
                        item["content"][:750],
                        f"=== KAYNAK [{index}] BİTİŞ ===",
                    ]
                )
            )

        prompt = (
            f"ARAMA KAPSAMI: {route['label']}\n\n"
            "BELGE BAĞLAMI\n\n"
            + "\n\n".join(context_blocks)
            + f"\n\nKULLANICI SORUSU\n{question}\n\n"
            "Yalnız yukarıdaki kanıtlardan, kısa Türkçe cevap ver."
        )
        answer = self.runtime.chat(
            RAG_SYSTEM_PROMPT,
            prompt,
            max_output_tokens=140,
        )
        route = dict(route)
        route["answer_mode"] = "Yerel modelle ayrıntılı yanıt"
        return answer, results, route

    def analyze_match(self, db_path: Path, cv_source: str, job_source: str) -> dict:
        cv_text = get_source_text(db_path, cv_source)
        job_text = get_source_text(db_path, job_source)
        result = self._guarded_match_from_texts(cv_text, job_text)
        result["cv_source"] = cv_source
        result["job_source"] = job_source
        return result
