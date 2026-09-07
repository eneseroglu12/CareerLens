from __future__ import annotations

import re


def _normalize(text: str) -> str:
    # Map Turkish capital dotted İ before lower(); otherwise Python produces
    # ``i`` + COMBINING DOT ABOVE and simple substring checks can fail.
    text = text.replace("İ", "I")
    folded = text.lower().translate(
        str.maketrans({"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"})
    )
    folded = folded.replace("\u0307", "")
    return re.sub(r"\s+", " ", folded).strip()


def route_question(question: str) -> dict:
    """Kariyer sorusu için en küçük ve güvenli belge kapsamını belirler.

    Yönlendirme LLM kullanmaz; deterministik ve yereldir. Böylece örneğin CV
    hakkındaki bir soru, embedding skoru yüksek diye kariyer rehberi veya iş ilanı
    parçalarıyla kirlenmez.
    """

    q = _normalize(question)

    cv_patterns = [
        r"\bcv\b", r"\bresume\b", r"\bcandidate\b", r"\bapplicant\b",
        r"my skills?", r"my experience", r"my projects?", r"my education",
        r"ozgecmis", r"aday", r"becerilerim", r"yetkinliklerim",
        r"deneyimlerim", r"projelerim", r"egitimim",
    ]
    job_patterns = [
        r"\bjob description\b", r"\bjob\b", r"\brole\b", r"\bposition\b",
        r"\brequirements?\b", r"\bresponsibilit", r"\bemployer\b",
        r"is ilani", r"ilan", r"pozisyon", r"gereksinim", r"gereklilik",
        r"sorumluluk", r"rol icin",
    ]
    guide_patterns = [
        r"career guide", r"career advice", r"best practice", r"tailor(?:ing)?",
        r"kariyer rehberi", r"kariyer tavsi", r"genel cv tavsi",
        r"iyi bir cv nasil", r"cv nasil yazilir", r"cv nasil hazirlanir",
        r"ozgecmis nasil yazilir", r"ozgecmis nasil hazirlanir",
    ]
    comparison_patterns = [
        r"\bmatch\b", r"\bfit\b", r"compare", r"suitable", r"qualified",
        r"gap", r"not evidenced", r"missing", r"skills?.*missing", r"meets? .*requirements?",
        r"uygun", r"uyuyor", r"uyum", r"esles", r"karsilastir", r"eksik", r"ne kadar uyum",
        r"bu role uygun", r"bu ilana uygun", r"karsiliyor mu",
    ]

    def has_any(patterns: list[str]) -> bool:
        return any(re.search(pattern, q) for pattern in patterns)

    has_cv = has_any(cv_patterns)
    has_job = has_any(job_patterns)
    has_guide = has_any(guide_patterns)
    has_comparison = has_any(comparison_patterns)
    candidate_specific = has_any(
        [
            r"\bcandidate\b", r"\bapplicant\b", r"my cv", r"my resume",
            r"this cv", r"this resume", r"aday", r"cv'm", r"cv(?:'|’)?de",
            r"bu cv", r"ozgecmisim", r"ozgecmis(?:im)?de",
            r"becerilerim", r"yetkinliklerim", r"deneyimlerim", r"projelerim",
        ]
    )

    # "CV'de neler geliştirilmeli?" gibi sorular mevcut CV'nin içeriğini
    # değerlendirmeyi ister. "iyileştir/geliştir" kelimesi tek başına kariyer
    # rehberi kapsamına yönlendirme sebebi değildir.
    cv_improvement_specific = has_cv and candidate_specific and has_any(
        [r"iyilestir", r"gelistir", r"guclendir", r"eksik", r"ne eklen"]
    )
    if cv_improvement_specific and not has_job and not has_guide:
        return {
            "doc_types": ["cv"],
            "label": "Yalnız CV",
            "reason": "Soru, yüklenen CV'nin hangi yönlerinin geliştirilebileceğini soruyor.",
            "comparison": False,
        }

    if has_comparison and (
        has_cv or has_job or "this role" in q or "bu rol" in q or "bu pozisyon" in q or "bu ilan" in q
    ):
        return {
            "doc_types": ["cv", "job"],
            "label": "CV + İş İlanı",
            "reason": "Soru, adayın CV kanıtlarını hedef pozisyonun gereksinimleriyle karşılaştırıyor.",
            "comparison": True,
        }

    if has_cv and has_job:
        return {
            "doc_types": ["cv", "job"],
            "label": "CV + İş İlanı",
            "reason": "Soru hem CV/adaya hem de iş ilanı/pozisyona açıkça referans veriyor.",
            "comparison": has_comparison,
        }

    if has_cv and has_guide:
        if candidate_specific:
            return {
                "doc_types": ["cv", "guide"],
                "label": "CV + Kariyer Rehberi",
                "reason": "Soru, kariyer rehberini adayın CV'sine göre uyarlamayı gerektiriyor.",
                "comparison": False,
            }
        return {
            "doc_types": ["guide"],
            "label": "Yalnız Kariyer Rehberi",
            "reason": "Soru adayın özel bilgileri yerine genel CV/kariyer rehberliği istiyor.",
            "comparison": False,
        }

    if has_cv:
        return {
            "doc_types": ["cv"],
            "label": "Yalnız CV",
            "reason": "Soru adayın veya CV'nin içindeki açık bilgilerle ilgili.",
            "comparison": False,
        }

    if has_job:
        return {
            "doc_types": ["job"],
            "label": "Yalnız İş İlanı",
            "reason": "Soru pozisyonun gereksinimleri veya sorumluluklarıyla ilgili.",
            "comparison": False,
        }

    if has_guide:
        return {
            "doc_types": ["guide"],
            "label": "Yalnız Kariyer Rehberi",
            "reason": "Soru genel kariyer veya CV rehberliği istiyor.",
            "comparison": False,
        }

    return {
        "doc_types": [],
        "label": "Tüm Belgeler",
        "reason": "Belge türü güvenilir biçimde belirlenemediği için tüm indekslenmiş belgelerde arama yapılıyor.",
        "comparison": False,
    }
