from __future__ import annotations

import re
import unicodedata


def _fold(text: str) -> str:
    lowered = unicodedata.normalize("NFKD", (text or "").lower())
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    return lowered.translate(str.maketrans({"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()


def redact_contacts(text: str) -> str:
    """Remove contact/identity noise from evidence shown in the UI."""
    cleaned = text or ""
    patterns = [
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"(?:https?://)?(?:www\.)?linkedin\.com/\S+",
        r"(?:https?://)?(?:www\.)?github\.com/\S+",
        r"\+?\d[\d\s().-]{8,}\d",
        r"mailto:[^\s)]+",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[•|]\s*", " · ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-;:·")
    return cleaned


def keyword_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9+#]{2,}", (text or "").lower())
    stop = {
        "ve", "veya", "ile", "bir", "olan", "olarak", "bu", "için", "icin", "the", "and",
        "with", "for", "from", "that", "have", "has", "your", "ilgili", "gibi", "daha", "çok",
        "cv", "ilan", "iş", "is", "olmak", "sahip", "gerek", "gereksinim", "nitelik", "minimum",
    }
    return {tok for tok in tokens if tok not in stop}


def _looks_like_fragment(text: str) -> bool:
    text = normalize_text(text)
    if not text or len(text) < 14 or len(text.split()) < 2:
        return True
    # Obvious chopped endings such as "Yönetim Bilişim Sistemleri 4."
    if re.search(r"\b\d+[.]?$", text) and len(text.split()) <= 5:
        return True
    if re.search(r"\b(page|sayfa)\b", text, re.IGNORECASE):
        return True
    return False


_SECTION_MARKERS = [
    "PROFESYONEL ÖZET", "DENEYİM", "EĞİTİM", "PROJELER", "BECERİLER", "BECERILER",
    "TEKNİK BECERİLER", "TEKNIK BECERILER", "YETKİNLİKLER", "YETKINLIKLER",
    "DİLLER", "DILLER", "KULÜP AKTİVİTELERİ", "KULUP AKTIVITELERI",
    "TEKNOLOJİLER", "TEKNOLOJILER", "GANO", "GPA",
]


def sentence_candidates(text: str) -> list[str]:
    """Recover short readable evidence snippets from flattened PDF text."""
    text = normalize_text(redact_contacts(text))
    if not text:
        return []

    # Re-create boundaries that PDF extraction often flattens.
    for marker in sorted(_SECTION_MARKERS, key=len, reverse=True):
        text = re.sub(rf"\s+(?={re.escape(marker)}\b)", "\n", text, flags=re.IGNORECASE)

    # Key/value fields can be glued together in one long line.
    text = re.sub(
        r"\s+(?=(?:Teknik Beceriler|Yetkinlikler|Diller|Teknolojiler|GANO|GPA)\s*:)",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    # Diller field should stop before unrelated date/role text.
    text = re.sub(
        r"((?:Diller|DİLLER)\s*:\s*[^\n]{2,80}?)(?=\s+\d{4}\s*[–-]\s*(?:\d{4}|Günümüz|Gunumu[zş]))",
        r"\1\n",
        text,
        flags=re.IGNORECASE,
    )
    # Skills should stop when another labelled field begins.
    text = re.sub(
        r"((?:Teknik Beceriler|Yetkinlikler|Teknolojiler)\s*:[^\n]+?)(?=\s+(?:Teknik Beceriler|Yetkinlikler|Diller|Teknolojiler)\s*:)",
        r"\1\n",
        text,
        flags=re.IGNORECASE,
    )

    raw_parts = re.split(r"(?<=[.!?])\s+|\n+|\s+[•]\s+|\s+·\s+|\s+-\s+", text)
    out: list[str] = []
    for raw in raw_parts:
        part = normalize_text(raw).strip("-–—• ")
        if not part:
            continue
        # Strip section label prefixes when they do not add evidence value.
        part = re.sub(r"^(?:PROFESYONEL ÖZET|DENEYİM|EĞİTİM|PROJELER|BECERİLER|BECERILER|KULÜP AKTİVİTELERİ|KULUP AKTIVITELERI)\s+", "", part, flags=re.IGNORECASE)
        if _looks_like_fragment(part):
            continue
        if len(part) > 175:
            part = part[:172].rstrip(" ,;:-") + "…"
        if part not in out:
            out.append(part)
    return out


# Each group is one logically distinct part of a requirement. A second evidence
# snippet is only useful if it covers a group the first snippet does not.
_ASPECT_RULES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("c1", "c2", "b2", "ingilizce", "english"), ("ingilizce", "english", "b2", "c1", "c2")),
    (("sql",), ("sql", "pl/sql")),
    (("excel",), ("excel",)),
    (("test", "yazılım test", "yazilim test"), ("test", "uat", "spec", "test plan", "test senaryo")),
    (("doküman", "dokuman", "documentation"), ("doküman", "dokuman", "spec", "test plan", "test senaryo", "fonksiyonel analiz")),
    (("bilgi paylaş", "bilgi paylas", "knowledge sharing"), ("bilgi paylaş", "bilgi paylas", "knowledge sharing")),
    (("müşteri", "musteri", "stakeholder", "customer"), ("müşteri", "musteri", "stakeholder", "customer", "iş ihtiyaç", "is ihtiyac", "iş gereksinim")),
    (("iletişim", "iletisim", "communication"), ("iletişim", "iletisim", "communication")),
    (("takım", "takim", "teamwork"), ("ekip", "takım", "takim", "teamwork", "koordinasyon")),
    (("sorumluluk", "responsibility"), ("sorumluluk", "görev dağılım", "gorev dagilim", "yönetim kurulu", "koordinasyon")),
    (("iş analizi", "is analizi", "business analysis"), ("iş analizi", "is analizi", "gereksinim analizi", "fonksiyonel analiz")),
    (("proje yönet", "proje yonet", "project management"), ("proje yönet", "proje yonet", "project management", "koordinasyon")),
    (("yazılım", "yazilim", "software"), ("yazılım", "yazilim", "flutter", "firebase", "dart", "python", "c#", "uygulama")),
    (("proje", "project"), ("proje", "project", "uygulama")),
    (("python",), ("python",)),
    (("pandas",), ("pandas",)),
    (("c#", "c sharp"), ("c#", "c sharp")),
    ((".net", "dotnet"), (".net", "dotnet", "asp.net")),
    (("rest api",), ("rest api", "api endpoint")),
    (("git",), ("git", "github")),
    (("asp.net",), ("asp.net core",)),
    (("entity framework",), ("entity framework",)),
    (("postgresql", "sql server"), ("postgresql", "postgres", "sql server")),
    (("powerpoint",), ("powerpoint",)),
    (("jira", "trello"), ("jira", "trello")),
    (("bpmn", "süreç haritalama", "surec haritalama"), ("bpmn", "süreç akışı", "surec akisi", "süreç haritalama", "surec haritalama")),
    (("raporlama", "reporting"), ("rapor", "raporlama", "kpi", "dashboard")),
    (("koordinasyon", "coordination"), ("koordinasyon", "koordine")),
    (("planlama", "planning"), ("planlama", "planladım", "planladim")),
    (("takip", "tracking"), ("takip", "takibini", "takip ettim")),
    (("makine öğren", "makine ogren", "machine learning"), ("scikit-learn", "sklearn", "tahmin modeli", "machine learning")),
    (("istatistik", "hipotez", "deney analizi"), ("istatistiksel analiz", "istatistiksel modelleme", "hipotez", "deney analizi")),
    (("öğren", "ogren", "gelişim", "gelisim", "motivation", "motive"), ("öğren", "ogren", "gelişim", "gelisim", "motive", "çözüm", "cozum")),
]


def _active_aspects(requirement_text: str) -> list[tuple[str, ...]]:
    req = _fold(requirement_text)
    groups: list[tuple[str, ...]] = []

    # Education requirements need two distinct signals: field and current/required
    # status. This prevents a generic "üniversite projesi" line from outranking
    # the actual degree line.
    edu_field_terms = (
        "yonetim bilisim sistemleri", "mis", "bilgisayar bilimleri", "bilgi teknolojileri",
        "muhendislik", "business analytics", "industrial engineering", "istatistik", "matematik",
        "bilgisayar muhendisligi", "yazilim muhendisligi", "isletme", "ekonomi",
    )
    if any(term in req for term in edu_field_terms):
        groups.append((
            "yönetim bilişim sistemleri", "yonetim bilisim sistemleri", "mis",
            "bilgisayar bilimleri", "bilgi teknolojileri", "mühendislik", "muhendislik",
            "business analytics", "industrial engineering", "istatistik", "matematik",
            "bilgisayar mühendisliği", "bilgisayar muhendisligi", "yazılım mühendisliği", "yazilim muhendisligi",
            "işletme", "isletme", "ekonomi",
        ))
    if any(term in req for term in ("mezun", "ogrenci", "lisans program", "degree", "graduate")):
        groups.append(("mezun", "öğrenci", "ogrenci", "lisans", "beklenen", "graduate", "student"))

    is_testing_requirement = "test" in req or "uat" in req
    for triggers, evidence_aliases in _ASPECT_RULES:
        # In phrases such as "yazılım test süreçleri", the word "yazılım"
        # describes testing, not a separate software-development requirement.
        if is_testing_requirement and triggers == ("yazılım", "yazilim", "software"):
            continue
        if any(_fold(trigger) in req for trigger in triggers):
            groups.append(evidence_aliases)
    return groups


def _covered_aspects(candidate: str, aspects: list[tuple[str, ...]]) -> set[int]:
    folded = _fold(candidate)
    covered: set[int] = set()
    for idx, aliases in enumerate(aspects):
        if any(_fold(alias) in folded for alias in aliases):
            covered.add(idx)
    return covered


def _semantic_hint_score(requirement_text: str, candidate: str) -> float:
    req_terms = keyword_set(requirement_text)
    cand_terms = keyword_set(candidate)
    direct = len(req_terms & cand_terms)
    aspects = _active_aspects(requirement_text)
    covered = len(_covered_aspects(candidate, aspects))
    # Ignore completely unrelated snippets. Brevity must never make an unrelated
    # line (for example a language line for a SQL requirement) score positively.
    if direct == 0 and covered == 0:
        return 0.0
    # Prefer concise direct evidence over broad skills dumps.
    brevity = max(0.0, 1.2 - len(candidate) / 150.0)
    # A concrete action/experience sentence is stronger UI evidence than a generic
    # keyword dump when both support the same requirement.
    req_folded = _fold(requirement_text)
    action_bonus = 1.5 if re.search(
        r"\b(hazirladim|kullandim|gelistirdim|destekledim|gorev aldim|calistim|inceledim|katki sagladim|hazirladi|kullandi|gelistirdi|destekledi)\b",
        _fold(candidate),
    ) else 0.0
    # For simple proficiency statements (SQL/English/Excel), a clean skills or
    # language line is preferable to a longer action sentence that only mentions
    # the term incidentally.
    if ("sql" in req_folded or "ingilizce" in req_folded or "english" in req_folded or "excel" in req_folded) and "test" not in req_folded:
        action_bonus = 0.0
    return direct * 2.0 + covered * 2.6 + brevity + action_bonus


def select_evidence_snippets(requirement_text: str, evidence_rows: list[dict], limit: int = 2) -> list[tuple[str, str]]:
    """Return one strong proof, plus a second only when it adds a new aspect."""
    ranked: list[tuple[float, str, str, set[int]]] = []
    aspects = _active_aspects(requirement_text)

    for row in evidence_rows:
        eid = str(row.get("evidence_id", ""))
        for candidate in sentence_candidates(str(row.get("text", ""))):
            score = _semantic_hint_score(requirement_text, candidate)
            coverage = _covered_aspects(candidate, aspects)
            if score <= 0:
                continue
            ranked.append((score, eid, candidate, coverage))

    ranked.sort(key=lambda item: (item[0], -len(item[2])), reverse=True)
    if not ranked:
        return []

    primary = ranked[0]
    chosen = [(primary[1], primary[2])]
    if limit <= 1:
        return chosen

    # If the strongest sentence already covers every detected aspect, more proof
    # only adds clutter. For single-aspect requirements we also keep one sentence.
    primary_coverage = set(primary[3])
    if len(aspects) <= 1 or (aspects and len(primary_coverage) == len(aspects)):
        return chosen

    primary_words = set(_fold(primary[2]).split())
    for score, eid, candidate, coverage in ranked[1:]:
        new_aspects = coverage - primary_coverage
        if not new_aspects:
            continue
        words = set(_fold(candidate).split())
        similarity = len(primary_words & words) / max(len(primary_words | words), 1)
        if similarity > 0.72:
            continue
        # Require a meaningful second proof; weak generic lines are discarded.
        if score < 2.2:
            continue
        chosen.append((eid, candidate))
        break
    return chosen[:limit]
