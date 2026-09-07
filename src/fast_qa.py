from __future__ import annotations

import re
from pathlib import Path

from .database import get_source_text
from .query_router import _normalize
from .requirement_guard import extract_cv_evidence_units, extract_job_requirements
from .ui_evidence import redact_contacts

TECH_TERMS = [
    "Python", "pandas", "NumPy", "SQL", "Microsoft Excel", "Excel", "Power BI", "Tableau",
    "Jira", "Trello", "BPMN", "UAT", "C#", ".NET", "ASP.NET Core", "Entity Framework Core",
    "PostgreSQL", "SQL Server", "REST API", "Git", "GitHub", "Flutter", "Firebase", "Dart",
    "scikit-learn", "Docker", "Java", "JavaScript", "TypeScript", "React", "Angular", "Azure",
]


def _clean(text: str) -> str:
    text = redact_contacts(text or "")
    text = re.sub(r"\s+", " ", text).strip(" \t\n-•")
    return text


def _source_from_results(results: list[dict], preferred_doc_type: str | None = None) -> str | None:
    if not results:
        return None
    if preferred_doc_type:
        for item in results:
            if item.get("doc_type") == preferred_doc_type:
                return str(item.get("source"))
    return str(results[0].get("source"))


def _full_text(db_path: Path, results: list[dict], preferred_doc_type: str | None = None) -> tuple[str, str | None]:
    source = _source_from_results(results, preferred_doc_type)
    if not source:
        return "", None
    return get_source_text(db_path, source, max_chars=18000), source


def _extract_tech_terms(text: str) -> list[str]:
    low = text.casefold()
    found: list[str] = []
    for term in TECH_TERMS:
        token = term.casefold()
        # Avoid reporting both "Excel" and "Microsoft Excel" when the long form exists.
        if token in low and term not in found:
            if term == "Excel" and "Microsoft Excel" in found:
                continue
            found.append(term)
    if "Microsoft Excel" in found and "Excel" in found:
        found.remove("Excel")
    return found


def _cefr(text: str) -> str | None:
    match = re.search(r"(?:ingilizce|english)\s*(?:[–—:\-]|seviyesi)?\s*(A1|A2|B1|B2|C1|C2)\b", text, re.I)
    if match:
        return match.group(1).upper()
    # Sometimes a line is flattened as "DİLLER İngilizce – C1".
    match = re.search(r"\bingilizce\b.{0,35}\b(A1|A2|B1|B2|C1|C2)\b", text, re.I)
    return match.group(1).upper() if match else None


def _relevant_units(text: str, keywords: tuple[str, ...], limit: int = 4) -> list[str]:
    units = extract_cv_evidence_units(text)
    out: list[str] = []
    for unit in units:
        n = _normalize(unit.text)
        if any(_normalize(k) in n for k in keywords):
            cleaned = _clean(unit.text)
            if cleaned and cleaned not in out:
                out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _best_action_units(text: str, limit: int = 3) -> list[str]:
    units = extract_cv_evidence_units(text)
    verbs = (
        "hazirlad", "gelistir", "analiz", "kulland", "olustur", "takip", "kontrol", "incele",
        "destek", "yonet", "koordine", "uygula", "gerceklestir", "sorgu", "rapor",
    )
    ranked: list[tuple[int, str]] = []
    for unit in units:
        cleaned = _clean(unit.text)
        n = _normalize(cleaned)
        if len(cleaned) < 30:
            continue
        score = sum(1 for verb in verbs if verb in n)
        score += sum(1 for term in TECH_TERMS if term.casefold() in cleaned.casefold())
        if score:
            ranked.append((score, cleaned))
    ranked.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    out: list[str] = []
    for _, text_item in ranked:
        if text_item not in out:
            out.append(text_item)
        if len(out) >= limit:
            break
    return out


def _cv_improvement_answer(text: str) -> str:
    units = extract_cv_evidence_units(text)
    if not units:
        return "CV metninde değerlendirebileceğim yeterli açık içerik bulamadım."

    suggestions: list[str] = []
    all_text = " ".join(u.text for u in units)
    n_all = _normalize(all_text)

    # Quantified impact: years/GPA alone do not count as an outcome metric.
    impact_pattern = re.compile(
        r"(?:%\s*\d+|\d+\s*(?:kullanici|musteri|islem|rapor|proje|test|kayit|saat|gun|hafta)|"
        r"artir|azalt|hizlandir|tasarruf|verimlilik|basari)",
        re.I,
    )
    if not impact_pattern.search(n_all):
        suggestions.append(
            "**Ölçülebilir etki ekle:** Deneyim maddeleri ne yaptığını gösteriyor; mümkünse sonuçları sayı, oran, süre veya hacimle desteklemek CV'yi güçlendirir."
        )

    skill_units = [u.text for u in units if u.text.count(",") >= 3]
    if skill_units:
        skill_text = " ".join(skill_units)
        listed = _extract_tech_terms(skill_text)
        used_elsewhere: list[str] = []
        non_skill_text = " ".join(u.text for u in units if u.text not in skill_units)
        for term in listed:
            if term.casefold() in non_skill_text.casefold():
                used_elsewhere.append(term)
                continue
            if term == "Microsoft Excel" and "excel" in non_skill_text.casefold():
                used_elsewhere.append(term)
        unsupported = [x for x in listed if x not in used_elsewhere]
        if unsupported:
            sample = ", ".join(unsupported[:3])
            suggestions.append(
                f"**Becerileri deneyimle kanıtla:** Beceriler bölümünde geçen {sample} gibi araçları gerçekten kullandıysan, bunları bir proje veya deneyim cümlesinde nasıl kullandığını belirt."
            )

    if not any(k in n_all for k in ("kullanici", "musteri", "stakeholder", "is birimi", "ekip")):
        suggestions.append(
            "**İş birliği bağlamını görünür yap:** Gerçekten yaptıysan ekip, kullanıcı, müşteri veya iş birimiyle nasıl çalıştığını bir deneyim maddesinde açıkça belirt."
        )

    # If the CV already has rich action evidence, praise it briefly and focus on polish.
    action_units = _best_action_units(text, limit=4)
    if action_units and len(suggestions) < 3:
        suggestions.append(
            "**Güçlü deneyim maddelerini öne taşı:** Somut eylem içeren maddeler mevcut; hedef role en yakın 3–4 maddeyi deneyim bölümünün üst sıralarında tut."
        )

    if not suggestions:
        suggestions.append(
            "CV'de açık deneyim ve beceri kanıtları mevcut. En büyük kazanım, hedef role göre en ilgili maddeleri öne çıkarıp metni daha kısa ve sonuç odaklı tutmak olur."
        )

    return "Belgedeki açık içeriklere göre en belirgin geliştirme alanları:\n\n" + "\n".join(
        f"- {item}" for item in suggestions[:4]
    )


def _cv_answer(question: str, text: str) -> str | None:
    q = _normalize(question)

    if any(k in q for k in ("gelistir", "iyilestir", "ne eklen", "eksik")):
        return _cv_improvement_answer(text)

    if any(k in q for k in ("teknik beceri", "teknoloji", "skills", "araclar", "becerileri listele")):
        skills = _extract_tech_terms(text)
        if skills:
            return "CV'de açıkça geçen teknik beceriler:\n\n- " + "\n- ".join(skills)

    if "ingiliz" in q or "english" in q:
        level = _cefr(text)
        if level:
            return f"CV'de İngilizce seviyesi **{level}** olarak belirtilmiş."
        return "CV'de açık bir İngilizce seviye bilgisi bulamadım."

    if "sql" in q:
        rows = _relevant_units(text, ("sql",), limit=5)
        action_words = ("kulland", "sorgu", "kontrol", "analiz", "temiz", "dogrula", "cekt", "incele")
        action_rows = [
            r for r in rows
            if r.count(",") <= 2 and any(w in _normalize(r) for w in action_words)
        ]
        rows = (action_rows or rows)[:3]
        if rows:
            return "Evet. CV'de SQL için açık kanıt var:\n\n" + "\n".join(f"- {r}" for r in rows)
        return "CV'de SQL kullanımını açıkça doğrulayan bir kanıt bulamadım."

    if any(k in q for k in ("yazilim test", "test deney", "uat", "test surec")):
        rows = _relevant_units(text, ("test", "uat"), limit=3)
        if rows:
            return "CV'de yazılım testi/UAT için açık kanıtlar:\n\n" + "\n".join(f"- {r}" for r in rows)
        return "CV'de yazılım testi veya UAT deneyimini açıkça doğrulayan bir kanıt bulamadım."

    if "proje yonet" in q:
        rows = _relevant_units(text, ("jira", "proje", "takip", "koordinasyon", "gorev"), limit=4)
        if rows:
            return (
                "CV'de proje yönetimiyle ilişkili kanıtlar var; ancak bunlar doğrudan bir 'proje yöneticisi' rolü anlamına gelmez:\n\n"
                + "\n".join(f"- {r}" for r in rows)
            )
        return "CV'de proje yönetimini açıkça doğrulayan bir deneyim bulamadım."

    if any(k in q for k in ("en guclu", "guclu uc", "guclu 3", "en iyi deney")):
        rows = _best_action_units(text, limit=3)
        if rows:
            return "CV'de en güçlü görünen somut deneyim kanıtları:\n\n" + "\n".join(f"- {r}" for r in rows)

    if any(k in q for k in ("deneyim", "staj")):
        rows = _best_action_units(text, limit=4)
        if rows:
            return "CV'deki öne çıkan deneyim kanıtları:\n\n" + "\n".join(f"- {r}" for r in rows)

    if any(k in q for k in ("egitim", "universite", "bolum")):
        rows = _relevant_units(text, ("universite", "lisans", "ogrenci", "muhendisligi", "yonetim bilisim", "istatistik", "isletme"), limit=3)
        if rows:
            return "CV'deki eğitim bilgileri:\n\n" + "\n".join(f"- {r}" for r in rows)

    return None


def _job_section(text: str, names: tuple[str, ...]) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip(" -•\t") for line in text.splitlines()]
    wanted = False
    out: list[str] = []
    headings = (
        "ARANAN NİTELİKLER", "ZORUNLU", "GEREKSİNİMLER", "TERCİH EDİLEN NİTELİKLER",
        "TERCİH EDİLEN", "GÖREV VE SORUMLULUKLAR", "SORUMLULUKLAR", "GÖREVLER",
    )
    for line in lines:
        upper = line.upper()
        if any(name in upper for name in names):
            wanted = True
            continue
        if wanted and any(h in upper for h in headings):
            break
        if wanted and line:
            out.append(line)
    return out


def _job_answer(question: str, text: str) -> str | None:
    q = _normalize(question)
    requirements = extract_job_requirements(text)

    if any(k in q for k in ("tercih", "arti nitelik", "nice to have")):
        rows = [r.text for r in requirements if r.category == "preferred"]
        if rows:
            return "Tercih edilen / artı nitelikler:\n\n" + "\n".join(f"- {r}" for r in rows)

    if any(k in q for k in ("zorunlu", "gereksinim", "aranan nitelik", "requirements")):
        rows = [r.text for r in requirements if r.category == "required"]
        if rows:
            return "İlandaki zorunlu gereksinimler:\n\n" + "\n".join(f"- {r}" for r in rows)

    if any(k in q for k in ("gorev", "sorumluluk")):
        rows = _job_section(text, ("GÖREV VE SORUMLULUKLAR", "SORUMLULUKLAR", "GÖREVLER"))
        if rows:
            return "İlandaki görev ve sorumluluklar:\n\n" + "\n".join(f"- {r}" for r in rows[:8])

    if any(k in q for k in ("teknik arac", "teknoloji", "hangi arac", "hangi teknoloji")):
        skills = _extract_tech_terms(text)
        if skills:
            return "İlanda açıkça geçen teknik araç/teknolojiler:\n\n- " + "\n- ".join(skills)

    if "ingiliz" in q or "english" in q:
        level = _cefr(text)
        if level:
            return f"İlanda İngilizce için açık seviye **{level}** olarak belirtilmiş."
        if "ingiliz" in _normalize(text) or "english" in _normalize(text):
            return "İlanda İngilizce bilgisi isteniyor; ancak açık bir CEFR seviyesi (B1/B2/C1 gibi) belirtilmemiş."

    return None


def _fallback_extract(question: str, results: list[dict]) -> str:
    q_words = {w for w in re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9+#.]+", _normalize(question)) if len(w) >= 3}
    candidates: list[tuple[int, str]] = []
    for item in results:
        content = _clean(str(item.get("content", "")))
        # Chunk text may be flattened. Break on punctuation first, then trim.
        parts = re.split(r"(?<=[.!?])\s+|\s+-\s+", content)
        for part in parts:
            part = _clean(part)
            if len(part) < 25:
                continue
            n = _normalize(part)
            score = sum(1 for w in q_words if w in n)
            candidates.append((score, part[:260]))
    candidates.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
    chosen: list[str] = []
    for _, part in candidates:
        if part not in chosen:
            chosen.append(part)
        if len(chosen) >= 3:
            break
    if not chosen:
        return "Seçilen belge kapsamında soruyla doğrudan ilişkilendirilebilecek açık bir kanıt bulamadım."
    return "Belgede soruyla en ilişkili açık kanıtlar:\n\n" + "\n".join(f"- {c}" for c in chosen)


def build_fast_answer(question: str, db_path: Path, results: list[dict], route: dict) -> tuple[str, str | None]:
    """Create a deterministic, near-instant answer without loading a chat model.

    Returns (answer, selected_source). The model-based answer remains available as
    an optional detail action in the UI, but the main Q&A path is always fast.
    """
    doc_types = route.get("doc_types") or []
    preferred = doc_types[0] if len(doc_types) == 1 else None
    text, source = _full_text(db_path, results, preferred)

    if preferred == "cv" and text:
        answer = _cv_answer(question, text)
        if answer:
            return answer, source
    elif preferred == "job" and text:
        answer = _job_answer(question, text)
        if answer:
            return answer, source
    elif preferred == "guide" and text:
        return _fallback_extract(question, results), source

    # Automatic routing can report [] for an ambiguous question. Use the source
    # type of the best hit to choose a safe deterministic answer.
    if results:
        best_type = str(results[0].get("doc_type", ""))
        if best_type == "cv" and text:
            answer = _cv_answer(question, text)
            if answer:
                return answer, source
        if best_type == "job" and text:
            answer = _job_answer(question, text)
            if answer:
                return answer, source

    return _fallback_extract(question, results), source
