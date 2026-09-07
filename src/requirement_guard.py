from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import numpy as np


STATUS_MATCHED = "matched"
STATUS_PARTIAL = "partial"
STATUS_NOT_EVIDENCED = "not_evidenced"
ALLOWED_STATUSES = {STATUS_MATCHED, STATUS_PARTIAL, STATUS_NOT_EVIDENCED}


@dataclass(frozen=True)
class Requirement:
    id: str
    text: str
    category: str  # required | preferred


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    text: str


_HEADING_REQUIRED = {
    "requirements",
    "qualifications",
    "required qualifications",
    "minimum qualifications",
    "what we are looking for",
    "what we're looking for",
    "you have",
    "aranan nitelikler",
    "aranan özellikler",
    "gereksinimler",
    "gereklilikler",
    "zorunlu nitelikler",
    "minimum nitelikler",
    "kimleri arıyoruz",
    "senden beklediklerimiz",
}

_HEADING_PREFERRED = {
    "nice to have",
    "preferred qualifications",
    "preferred",
    "bonus",
    "good to have",
    "tercih edilen nitelikler",
    "tercihen",
    "artı nitelikler",
    "ek nitelikler",
}

_SECTION_BREAKS = {
    "responsibilities",
    "what you will do",
    "what you'll do",
    "about the role",
    "about us",
    "benefits",
    "what we offer",
    "responsibility",
    "sorumluluklar",
    "görev ve sorumluluklar",
    "gorev ve sorumluluklar",
    "duties and responsibilities",
    "görevler",
    "gorevler",
    "iş tanımı",
    "is tanimi",
    "rol hakkında",
    "rol hakkinda",
    "hakkımızda",
    "hakkimizda",
    "yan haklar",
    "sunduklarımız",
    "sunduklarimiz",
}

_REQUIREMENT_CUES = re.compile(
    r"\b(required|requirement|must|should|knowledge of|experience with|experience in|"
    r"familiarity with|proficiency|proficient|ability to|skills?|student|graduate|degree|"
    r"english|turkish|minimum|at least|years? of|aranan|gerekl|required|bilgi sahibi|"
    r"deneyim|hakim|yetkin|öğrenci|ogrenci|mezun|ingilizce|en az)\b",
    re.IGNORECASE,
)

_BULLET_RE = re.compile(r"^\s*(?:[-*•▪◦‣–—]|\d+[.)])\s+")


def _fold(text: str) -> str:
    # Python lowercases capital Turkish İ to ``i`` + COMBINING DOT ABOVE.
    # Removing combining marks first makes İstanbul/İngilizce/İş Analitiği
    # normalize consistently with lowercase aliases and prevents false negatives.
    lowered = unicodedata.normalize("NFKD", text.lower())
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    return (
        lowered
        .translate(str.maketrans({"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}))
        .strip(" :.-\t")
    )


def _clean_line(line: str) -> str:
    line = line.replace("\u00a0", " ").strip()
    line = _BULLET_RE.sub("", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _heading_kind_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for heading in _HEADING_REQUIRED:
        out[_fold(heading)] = "required"
    for heading in _HEADING_PREFERRED:
        out[_fold(heading)] = "preferred"
    for heading in _SECTION_BREAKS:
        out[_fold(heading)] = "break"
    return out


def _is_heading(line: str) -> bool:
    clean = _clean_line(line)
    if not clean or len(clean) > 90:
        return False
    # "Diller: İngilizce – B2" veya "Beceriler: SQL, Excel" gibi
    # kısa CV satırları içerik taşır; başlık sanılıp kanıttan atılmamalıdır.
    if ":" in clean and clean.split(":", 1)[1].strip():
        return False
    folded = _fold(clean)
    # Degree/student identity lines are evidence, not section headings.
    if any(term in folded for term in ("ogrenci", "student", "lisans", "mezun", "graduate")) and any(
        term in folded for term in (
            "yonetim bilisim sistemleri", "istatistik", "matematik", "bilgisayar muhendisligi",
            "yazilim muhendisligi", "endustri muhendisligi", "isletme", "ekonomi", "business analytics"
        )
    ):
        return False
    if folded in _heading_kind_map():
        return True
    # Short CV lines may look like headings while carrying critical evidence
    # ("MIS ÖĞRENCİSİ", "... – Lisans", "İngilizce – B2"). Keep them.
    if re.search(
        r"\b(lisans|ogrenci|mezun|mis|yonetim bilisim sistemleri|sql|excel|ingilizce|b2|c1|c2|stajyer|intern|analist|test|power bi|jira|agile)\b",
        folded,
    ):
        return False
    # Kısa, noktalamasız başlıkları da CV kanıt birimlerinden ayır.
    if len(clean.split()) <= 7 and not re.search(r"[.!?]$", clean):
        if clean.isupper() or clean.istitle():
            return True
    return False


def _heading_regex(heading: str) -> re.Pattern[str]:
    # Başlık içindeki boşluklar esnektir. Başlığın gerçek bir bölüm başlığı
    # sayılması için ardından iki nokta, satır sonu ya da madde ayıracı gelmesi
    # gerekir. Böylece "business requirements from stakeholders" içindeki
    # requirements kelimesi yanlışlıkla bölüm başlığı olarak yakalanmaz.
    escaped = re.escape(heading)
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(
        rf"(?<!\w){escaped}\s*(?P<sep>:|(?:\r?\n)+|[-–—•▪◦‣]\s+)",
        re.IGNORECASE,
    )


def _find_section_markers(text: str) -> list[tuple[int, int, str, str]]:
    markers: list[tuple[int, int, str, str]] = []
    headings = [
        *((heading, "required") for heading in _HEADING_REQUIRED),
        *((heading, "preferred") for heading in _HEADING_PREFERRED),
        *((heading, "break") for heading in _SECTION_BREAKS),
    ]
    # Aynı konumda "preferred" ve "preferred qualifications" gibi örtüşmeler
    # olursa önce uzun başlığı tut.
    headings.sort(key=lambda pair: len(pair[0]), reverse=True)
    for heading, kind in headings:
        for match in _heading_regex(heading).finditer(text):
            markers.append((match.start(), match.end(), kind, heading))

    markers.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    deduped: list[tuple[int, int, str, str]] = []
    for marker in markers:
        if deduped and marker[0] < deduped[-1][1]:
            continue
        deduped.append(marker)
    return deduped


def _split_section_items(section_text: str) -> list[str]:
    """Bölüm metnini satır sonları kaybolmuş olsa bile ayrı maddelere böler."""
    text = section_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    # Satır başındaki ya da düzleştirilmiş metindeki " - " / " • " maddeleri
    # yapısal bir ayıraca çevir. Kelime içi tireler (future-state, problem-solving)
    # etkilenmez çünkü iki yanında boşluk şartı vardır.
    text = re.sub(r"(?m)^\s*[-*•▪◦‣–—]\s+", "\n§ ", text)
    text = re.sub(r"\s+[-*•▪◦‣–—]\s+", "\n§ ", text)
    text = re.sub(r"\n+", "\n", text)

    raw_parts = [part.strip() for part in re.split(r"\n§\s*|\n+", text) if part.strip()]
    parts: list[str] = []
    for part in raw_parts:
        clean = _clean_line(part)
        if not clean:
            continue
        # Madde ayıracı olmayan uzun paragraflarda cümle sınırlarını yedek olarak
        # kullan. Normal, kısa nitelik maddeleri tek parça kalır.
        if len(clean) > 320:
            sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])", clean)
                if s.strip()
            ]
            if len(sentences) > 1:
                parts.extend(sentences)
                continue
        parts.append(clean)
    return parts


def _dedupe_requirement_texts(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = _clean_line(item)
        key = _fold(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def extracted_requirements_are_safe(requirements: list[Requirement], job_text: str) -> bool:
    """Parser'ın tek dev paragrafı yanlışlıkla tek gereksinim saymasını engeller."""
    if not requirements:
        return False
    texts = [req.text for req in requirements]
    if any(len(text) > 520 for text in texts):
        return False
    # Uzun bir ilan metninden yalnızca tek, aşırı uzun gereksinim çıkması önceki
    # bug'ın tipik imzasıdır; böyle durumda yanlış skor üretmek yerine dur.
    if len(requirements) == 1 and len(job_text) > 650 and len(texts[0]) > 220:
        return False
    heading_words = {
        _fold(x) for x in (_HEADING_REQUIRED | _HEADING_PREFERRED | _SECTION_BREAKS)
    }
    for item in texts:
        folded = _fold(item)
        # Aynı gereksinim maddesinde birkaç bölüm başlığının birlikte görünmesi
        # bölüm ayrıştırmasının kaçırıldığına işaret eder.
        hits = sum(1 for heading in heading_words if heading and heading in folded)
        if hits >= 2:
            return False
    return True


def extract_job_requirements(job_text: str, max_items: int = 20) -> list[Requirement]:
    """İş ilanındaki zorunlu ve tercih edilen nitelikleri güvenli biçimde çıkar.

    Hem düzgün satır sonları olan belgeleri hem de chunking/PDF çıkarımı nedeniyle
    tek satıra düzleşmiş şu formu destekler:
        Responsibilities - ... - ... Requirements - ... - ... Nice to have - ...

    Sorumluluklar skora dahil edilmez; yalnız açık gereksinim/nitelik bölümleri
    değerlendirilir. Başlık bulunamazsa muhafazakâr bir cue tabanlı fallback vardır.
    """
    source = job_text.replace("\u00a0", " ").strip()
    required: list[str] = []
    preferred: list[str] = []

    markers = _find_section_markers(source)
    saw_qualification_heading = any(kind in {"required", "preferred"} for _, _, kind, _ in markers)

    if saw_qualification_heading:
        for index, (_, end, kind, _) in enumerate(markers):
            next_start = markers[index + 1][0] if index + 1 < len(markers) else len(source)
            if kind not in {"required", "preferred"}:
                continue
            section = source[end:next_start].strip()
            items = _split_section_items(section)
            if kind == "required":
                required.extend(items)
            else:
                preferred.extend(items)
    else:
        # Yapısız ilanlar için satır/cümle bazlı ve muhafazakâr fallback.
        candidates = []
        for raw in source.splitlines():
            clean = _clean_line(raw)
            if clean:
                candidates.append(clean)
        if len(candidates) <= 1:
            candidates = [
                p.strip() for p in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", source))
                if p.strip()
            ]
        for clean in candidates:
            if _is_heading(clean):
                continue
            if _REQUIREMENT_CUES.search(_fold(clean)):
                required.append(clean)

    required = _dedupe_requirement_texts(required)
    preferred = _dedupe_requirement_texts(preferred)

    result: list[Requirement] = []
    counter = 1
    for item in required[:max_items]:
        result.append(Requirement(id=f"R{counter}", text=item, category="required"))
        counter += 1
    remaining = max(0, max_items - len(result))
    for item in preferred[:remaining]:
        result.append(Requirement(id=f"R{counter}", text=item, category="preferred"))
        counter += 1
    return result

def _contains_contact_like_data(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
        r"(?:https?://)?(?:www\.)?linkedin\.com/\S+",
        r"(?:https?://)?(?:www\.)?github\.com/\S+",
        r"\+?\d[\d\s().-]{8,}\d",
    ]
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns)


def _evidence_unit_quality_ok(text: str) -> bool:
    clean = _clean_line(text)
    folded = _fold(clean)
    meaningful_short = bool(re.search(
        r"\b(mis|ogrenci|mezun|lisans|sql|excel|jira|agile|uat|python|flutter|firebase|dart|power bi|ingilizce|b2|c1|c2|stajyer|analist)\b",
        folded,
    ))
    if len(clean) < 18 and not meaningful_short:
        return False
    if len(clean.split()) < 3 and not meaningful_short:
        return False
    if _contains_contact_like_data(clean):
        # Contact info may still be mixed into a useful long line; keep only if
        # there is substantial non-contact content as well.
        sanitized = re.sub(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", " ", clean, flags=re.IGNORECASE)
        sanitized = re.sub(r"(?:https?://)?(?:www\.)?linkedin\.com/\S+", " ", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"(?:https?://)?(?:www\.)?github\.com/\S+", " ", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\+?\d[\d\s().-]{8,}\d", " ", sanitized)
        sanitized = re.sub(r"\s+", " ", sanitized).strip(" .,-;:")
        if len(sanitized) < 24 or len(sanitized.split()) < 4:
            return False
    # Suppress obvious chopped endings such as "Yönetim Bilişim Sistemleri 4."
    # but keep normal short evidence sentences like "Toplantı notlarını hazırladım."
    if re.search(r"\b\d+[.]$", clean) and "sinif" not in folded:
        return False
    return True


def extract_cv_evidence_units(cv_text: str, max_units: int = 80) -> list[EvidenceUnit]:
    """Create short, auditable CV evidence units.

    Bullets and concise content lines are retained separately so the match model
    can refer to stable evidence IDs instead of inventing quotations.
    """

    units: list[str] = []
    for raw in cv_text.splitlines():
        clean = _clean_line(raw)
        if not clean or _is_heading(raw):
            continue
        if len(clean) < 3:
            continue
        # Split unusually long prose lines into sentences.
        if len(clean) > 240:
            # Do not split Turkish ordinal/class expressions such as "4. sınıf".
            # Sentence periods are accepted only after a letter; ! and ? are safe.
            parts = re.split(r"(?<=[!?])\s+|(?<=[A-Za-zÇĞİÖŞÜçğıöşü])\.\s+", clean)
            units.extend(part.strip() for part in parts if len(part.strip()) >= 3)
        else:
            units.append(clean)

    seen: set[str] = set()
    result: list[EvidenceUnit] = []
    for text in units:
        text = _clean_line(text)
        if not _evidence_unit_quality_ok(text):
            continue
        key = _fold(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(EvidenceUnit(id=f"E{len(result) + 1}", text=text))
        if len(result) >= max_units:
            break
    return result


def cosine_matrix(left: list[list[float]], right: list[list[float]]) -> np.ndarray:
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T


def _contains_alias(folded_text: str, alias: str) -> bool:
    alias_folded = _fold(alias)
    if not alias_folded:
        return False
    # Short technical tokens such as SQL/MIS/Jira should respect token boundaries.
    if re.fullmatch(r"[a-z0-9+#.]+", alias_folded):
        return re.search(rf"(?<![a-z0-9]){re.escape(alias_folded)}(?![a-z0-9])", folded_text) is not None
    return alias_folded in folded_text


_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "sql": ("sql",),
    "excel": ("excel",),
    "power_bi": ("power bi", "powerbi"),
    "dashboard": ("dashboard", "dashboarding", "gösterge paneli", "gosterge paneli", "tableau", "looker", "power bi", "powerbi"),
    "jira": ("jira",),
    "trello": ("trello",),
    "agile": ("agile", "çevik", "cevik"),
    "process_modeling": (
        "süreç modelleme", "surec modelleme", "süreç modellemesi", "surec modellemesi",
        "süreç haritalama", "surec haritalama", "iş süreçlerinin modellenmesi",
        "is sureclerinin modellenmesi", "process modeling", "process modelling",
        "process mapping", "business process modeling", "business process mapping",
        "süreç akışı", "surec akisi", "süreç akışları", "surec akislari", "bpmn",
    ),
    "software_testing": (
        "yazılım test", "yazilim test", "yazılım testi", "yazilim testi", "yazılım testine", "yazilim testine",
        "software testing", "software test", "uat", "kullanıcı kabul testi",
        "kullanici kabul testi", "test takibi", "hata kayıt", "hata kayit", "defect tracking",
        "test planlama", "test planning", "test planları", "test planlari", "test senaryoları", "test senaryolari",
    ),
    "analytical": ("analitik", "analytical"),
    "problem_solving": ("problem çözme", "problem cozme", "problem-solving", "problem solving"),
    "english": ("ingilizce", "english"),
    "written": ("yazılı", "yazili", "written"),
    "verbal": ("sözlü", "sozlu", "verbal", "oral"),
    "communication": (
        "iletişim", "iletisim", "communication", "toplantı", "toplanti",
        "teknik ekiplerle", "iş birimleriyle", "is birimleriyle", "ekipler arası", "ekipler arasi",
    ),
    "student": ("öğrenci", "ogrenci", "öğrencisi", "ogrencisi", "öğrenciyim", "ogrenciyim", "öğrencisiyim", "ogrencisiyim", "student"),
    "graduate": ("mezun", "graduate"),
    "business_analytics": ("iş analitiği", "is analitigi", "business analytics"),
    "mis": ("mis", "yönetim bilişim sistemleri", "yonetim bilisim sistemleri", "management information systems"),
    "industrial_engineering": ("endüstri mühendisliği", "endustri muhendisligi", "industrial engineering"),
    "business_analysis": (
        "iş analizi", "is analizi", "business analysis", "gereksinim analizi", "requirements analysis",
        "fonksiyonel analiz", "functional analysis",
    ),
    "project_management": ("proje yönetimi", "proje yonetimi", "project management"),
    "software_development": (
        "yazılım proje", "yazilim proje", "software project", "yazılım geliştirme", "yazilim gelistirme", "software development", "uygulama geliştirdim",
        "uygulama gelistirdim", "flutter", "firebase", "dart", "c#", "python",
    ),
    "project_experience": (
        "proje", "project", "kampüs uygulaması", "kampus uygulamasi", "uygulama geliştirdim", "uygulama gelistirdim",
    ),
    "customer_needs": (
        "müşteri ihtiyaç", "musteri ihtiyac", "müşteri gereksinim", "musteri gereksinim",
        "customer need", "client need", "client requirement", "customer requirement",
        "stakeholder need", "stakeholder requirement",
    ),
    "business_needs": (
        "iş ihtiyaç", "is ihtiyac", "iş gereksinim", "is gereksinim",
        "business need", "business requirement",
    ),
    "teamwork": (
        "takım çalışması", "takim calismasi", "teamwork", "ekip içi", "ekip ici", "ekiple", "ekip çalış", "ekip calis",
        "takım proje", "takim proje", "takım geliştirme", "takim gelistirme",
        "teknik ekiplerle", "iş birimleriyle", "is birimleriyle", "ekip toplantı", "ekip toplanti",
    ),
    "responsibility": (
        "sorumluluk", "responsibility", "görev dağılımı", "gorev dagilimi", "yönetim kurulu", "yonetim kurulu",
        "koordinasyon", "karar süreç", "karar surec",
    ),
    "documentation": (
        "dokümantasyon", "dokumantasyon", "dokümantasyona", "dokumantasyona",
        "doküman", "dokuman", "dokümanları", "dokumanlari", "documentation",
        "spec", "test plan", "test senaryo", "test scenario",
        "fonksiyonel analiz", "functional analysis", "user story", "acceptance criteria",
        "toplantı not", "toplanti not", "toplantılarının not", "toplantilarinin not",
        "notlarını hazır", "notlarini hazir", "meeting notes", "rapor hazırl", "rapor hazirla",
    ),
    "knowledge_sharing": ("bilgi paylaş", "bilgi paylas", "knowledge sharing", "knowledge transfer"),
    "learning_growth": (
        "öğrenmeye açık", "ogrenmeye acik", "gelişim odaklı", "gelisim odakli", "learning", "growth",
        "continuous learning", "geliştirmeye", "gelistirmeye",
    ),
    "solution_motivation": (
        "çözüm üret", "cozum uret", "motive", "motivasyon", "solution-oriented", "solution oriented",
    ),
    "statistics_degree": ("istatistik", "statistics"),
    "mathematics_degree": ("matematik", "mathematics"),
    "computer_engineering": ("bilgisayar mühendisliği", "bilgisayar muhendisligi", "computer engineering"),
    "software_engineering": ("yazılım mühendisliği", "yazilim muhendisligi", "software engineering"),
    "computer_science": ("bilgisayar bilimleri", "computer science"),
    "information_technology": ("bilgi teknolojileri", "information technology", "information technologies"),
    "business_degree": ("işletme", "isletme", "business administration", "business degree"),
    "economics_degree": ("ekonomi", "economics"),
    "python": ("python",),
    "pandas": ("pandas",),
    "csharp": ("c#", "c sharp"),
    "dotnet": (".net", "dotnet", "net ekosistemi"),
    "rest_api": ("rest api", "restful api", "api endpoint", "api geliştirme", "api gelistirme"),
    "git": ("git", "github"),
    "oop": ("nesne yönelimli", "nesne yonelimli", "object oriented", "object-oriented", "oop"),
    "aspnet_core": ("asp.net core", "aspnet core"),
    "entity_framework": ("entity framework core", "entity framework"),
    "postgresql": ("postgresql", "postgres"),
    "sql_server": ("sql server", "mssql"),
    "relational_database": ("ilişkisel veri tabanı", "iliskisel veri tabani", "relational database", "postgresql", "postgres", "sql server", "mssql"),
    "docker": ("docker",),
    "powerpoint": ("powerpoint", "power point"),
    "coordination": ("koordinasyon", "koordinasyonu", "koordinasyonunda", "coordination", "koordine"),
    "planning": ("planlama", "planning", "planladım", "planladim", "etkinlik planlama"),
    "tracking": ("takip", "tracking", "takip ettim", "takibini", "görev takibi", "gorev takibi"),
    "reporting": ("raporlama", "reporting", "rapor", "raporlar", "raporlarını", "raporlarini", "raporlarının", "raporlarinin", "operasyon rapor", "kpi rapor", "sunum"),
    "data_analysis": ("veri analizi", "data analysis", "keşifsel veri analizi", "kesifsel veri analizi", "exploratory data analysis", "eda"),
    "machine_learning": ("makine öğrenmesi", "makine ogrenmesi", "machine learning", "scikit-learn", "sklearn", "churn tahmin", "tahmin modeli"),
    "statistics_analysis": ("istatistiksel analiz", "istatistiksel modelleme", "statistics", "istatistik"),
    "hypothesis_testing": ("hipotez test", "hypothesis test", "a/b test", "ab test", "deney analizi", "experiment analysis"),
    "user_needs": ("kullanıcı ihtiyaç", "kullanici ihtiyac", "user need", "user requirement"),
    "club_activity": ("öğrenci kulüb", "ogrenci kulub", "kulüp", "kulup", "gönüllülük", "gonulluluk", "etkinlik koordinatörü", "etkinlik koordinatoru"),
}


def _concepts_in_text(text: str) -> set[str]:
    folded = _fold(text)
    concepts: set[str] = set()
    for concept, aliases in _CONCEPT_ALIASES.items():
        if any(_contains_alias(folded, alias) for alias in aliases):
            concepts.add(concept)
    return concepts


def _lexical_requirement_score(requirement_text: str, evidence_text: str) -> float:
    req_terms = _keyword_set(requirement_text)
    ev_terms = _keyword_set(evidence_text)
    token_overlap = len(req_terms & ev_terms) / max(len(req_terms), 1)
    req_concepts = _concepts_in_text(requirement_text)
    ev_concepts = _concepts_in_text(evidence_text)
    concept_overlap = len(req_concepts & ev_concepts) / max(len(req_concepts), 1) if req_concepts else 0.0
    return max(token_overlap, concept_overlap)


def build_candidate_evidence(
    requirements: list[Requirement],
    evidence_units: list[EvidenceUnit],
    requirement_vectors: list[list[float]],
    evidence_vectors: list[list[float]],
    top_n: int = 6,
) -> dict[str, list[dict]]:
    """Create an auditable semantic shortlist with deterministic lexical rescue.

    Exact career terms (SQL, Excel, process mapping, UAT, degree names, etc.) are
    allowed to rescue an evidence line even when the embedding ranking alone would
    place it just outside top-N. This prevents false negatives without inventing
    evidence: every returned row is still a literal CV evidence unit.
    """
    scores = cosine_matrix(requirement_vectors, evidence_vectors)
    candidates: dict[str, list[dict]] = {}
    for i, requirement in enumerate(requirements):
        if not evidence_units:
            candidates[requirement.id] = []
            continue

        ranked: list[tuple[float, int, float, float]] = []
        for j, evidence in enumerate(evidence_units):
            sim = float(scores[i, j])
            lexical = _lexical_requirement_score(requirement.text, evidence.text)
            # Semantic similarity remains the main retrieval signal. Explicit
            # lexical/concept overlap gets a strong but bounded rescue weight.
            combined = 0.68 * max(0.0, min(sim, 1.0)) + 0.32 * lexical
            ranked.append((combined, j, sim, lexical))

        ranked.sort(key=lambda item: (item[0], item[3], item[2]), reverse=True)
        selected = ranked[: max(1, top_n)]
        candidates[requirement.id] = [
            {
                "evidence_id": evidence_units[j].id,
                "text": evidence_units[j].text,
                "similarity": sim,
                "lexical_score": lexical,
            }
            for _, j, sim, lexical in selected
        ]
    return candidates

def build_candidate_evidence_fast(
    requirements: list[Requirement],
    evidence_units: list[EvidenceUnit],
    top_n: int = 12,
) -> dict[str, list[dict]]:
    """Build a fast shortlist of CV evidence for each job requirement.

    The shortlist uses Turkish-aware concept aliases and lexical overlap over the
    extracted CV evidence units. Foundry Local remains responsible for document
    embeddings and optional local-model answers.
    """
    candidates: dict[str, list[dict]] = {}
    for requirement in requirements:
        req_concepts = _concepts_in_text(requirement.text)
        ranked: list[tuple[float, EvidenceUnit]] = []
        for evidence in evidence_units:
            lexical = _lexical_requirement_score(requirement.text, evidence.text)
            ev_concepts = _concepts_in_text(evidence.text)
            shared = len(req_concepts & ev_concepts)
            concept_bonus = shared / max(len(req_concepts), 1) if req_concepts else 0.0
            score = max(lexical, concept_bonus)
            # Keep a tiny floor for general prose so conservative fallback can
            # still inspect a few semantically plausible text lines by token overlap.
            ranked.append((score, evidence))
        ranked.sort(key=lambda item: item[0], reverse=True)

        # Include the strongest rows plus every literal concept hit. This avoids
        # false negatives while remaining cheap for normal 1–2 page CVs.
        selected: list[EvidenceUnit] = [ev for _, ev in ranked[: max(1, top_n)]]
        for score, ev in ranked:
            if score <= 0:
                continue
            if ev not in selected:
                selected.append(ev)
        candidates[requirement.id] = [
            {
                "evidence_id": ev.id,
                "text": ev.text,
                "similarity": 0.0,
                "lexical_score": _lexical_requirement_score(requirement.text, ev.text),
            }
            for ev in selected
        ]
    return candidates


_CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


def _cefr_levels(text: str) -> list[str]:
    return re.findall(r"(?<![A-Z0-9])(A1|A2|B1|B2|C1|C2)(?![A-Z0-9])", text.upper())


def _best_cefr(texts: Iterable[str]) -> str | None:
    levels: list[str] = []
    for text in texts:
        levels.extend(_cefr_levels(text))
    return max(levels, key=lambda x: _CEFR_ORDER[x]) if levels else None


def classification_prompt(requirements: list[Requirement], candidates: dict[str, list[dict]]) -> str:
    blocks: list[str] = []
    for req in requirements:
        evidence_lines = candidates.get(req.id, [])
        evidence_text = "\n".join(
            f"- {item['evidence_id']}: {item['text']}" for item in evidence_lines
        ) or "- NONE"
        blocks.append(
            f"{req.id} [{req.category}] {req.text}\n"
            f"Allowed CV evidence for {req.id}:\n{evidence_text}"
        )

    return f"""Aşağıdaki iş gereksinimlerini TEK TEK değerlendir.

ZORUNLU KURALLAR:
- Yalnız her gereksinimin altında verilen 'Allowed CV evidence' satırlarını kullan.
- Bir gereksinimi atlama ve yeni gereksinim ekleme.
- status yalnızca şu üç değerden biri olabilir: matched, partial, not_evidenced.
- matched: kanıt gereksinimin ana unsurlarını açıkça karşılıyor.
- partial: ilişkili kanıt var ama gereksinimin tamamı açıkça doğrulanmıyor.
- not_evidenced: verilen kanıtlarda gereksinimi destekleyen yeterli açık kanıt yok.
- matched veya partial için evidence_ids boş olamaz ve yalnız o gereksinim altında izin verilen E kimliklerini kullanabilirsin.
- not_evidenced için evidence_ids boş liste olmalı.
- Sadece geçerli JSON döndür. Markdown veya ek açıklama yazma.

GEREKSİNİMLER VE İZİN VERİLEN CV KANITLARI:
{chr(10).join(chr(10) + block for block in blocks)}

JSON ŞEMASI:
[
  {{"requirement_id":"R1","status":"matched|partial|not_evidenced","evidence_ids":["E1"]}}
]
"""


def _extract_json_array(text: str) -> list[dict] | None:
    text = text.strip()
    # Remove common fenced-code wrappers.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, list) else None
        except Exception:
            return None
    return None


def _keyword_set(text: str) -> set[str]:
    folded = _fold(text)
    tokens = re.findall(r"[a-z0-9+#.]{2,}", folded)
    # Requirement boilerplate is excluded so explicit content terms (SQL, Excel,
    # process, testing, English, degree field) dominate the auditable overlap.
    stop = {
        "and", "or", "with", "the", "for", "to", "of", "in", "a", "an", "is", "are",
        "ve", "veya", "ile", "icin", "bir", "bu", "olan", "olarak", "en", "az", "ya", "da",
        "skills", "skill", "knowledge", "experience", "familiarity", "ability", "strong",
        "beceri", "becerileri", "becerilerine", "bilgi", "bilgisine", "sahip", "olmak",
        "konusunda", "calisma", "asina", "deneyim", "deneyime", "gereksinim", "nitelik",
    }
    return {tok for tok in tokens if tok not in stop}


def _evidence_ids_for_concept(candidate_rows: list[dict], concept: str) -> list[str]:
    return [
        row["evidence_id"]
        for row in candidate_rows
        if concept in _concepts_in_text(row["text"])
    ]


def _unique_ids(ids: Iterable[str], limit: int = 3) -> list[str]:
    out: list[str] = []
    for eid in ids:
        if eid not in out:
            out.append(eid)
        if len(out) >= limit:
            break
    return out


def explicit_requirement_status(
    requirement: Requirement, candidate_rows: list[dict]
) -> tuple[str | None, list[str]]:
    """High-confidence deterministic checks for common career requirements.

    This layer can correct both overclaims and false negatives from a small local
    LLM, but only when literal CV evidence supports named concepts. It never
    fabricates a skill. None means the requirement is left to the general guard.
    """
    req = _concepts_in_text(requirement.text)
    present: set[str] = set()
    for row in candidate_rows:
        present |= _concepts_in_text(row["text"])

    def ids_for(*concepts: str) -> list[str]:
        # Önce her kavram için en az bir kanıt göster; aksi halde ilk kavramın
        # birden fazla satırı 3 kanıtlık kotayı doldurup ikinci kavramın kanıtını
        # ekrandan saklayabiliyordu.
        primary: list[str] = []
        extras: list[str] = []
        for concept in concepts:
            hits = _evidence_ids_for_concept(candidate_rows, concept)
            if hits:
                primary.append(hits[0])
                extras.extend(hits[1:])
        return _unique_ids(primary + extras)

    # Education requirements: field alternatives are OR, but current status is
    # respected. A student does not fully satisfy a graduate-only requirement.
    degree_fields = req & {
        "business_analytics", "mis", "industrial_engineering", "statistics_degree",
        "mathematics_degree", "computer_engineering", "software_engineering",
        "computer_science", "information_technology", "business_degree", "economics_degree",
    }
    if degree_fields and ("student" in req or "graduate" in req):
        matched_fields = degree_fields & present
        wants_student = "student" in req
        wants_graduate = "graduate" in req
        has_student = "student" in present
        has_graduate = "graduate" in present
        if wants_student and wants_graduate:
            state_ok = has_student or has_graduate
            related_state = state_ok
        elif wants_graduate:
            state_ok = has_graduate
            related_state = has_student or has_graduate
        else:
            state_ok = has_student
            related_state = has_student or has_graduate
        if matched_fields and state_ok:
            return STATUS_MATCHED, ids_for(*(list(matched_fields) + (["graduate"] if wants_graduate and not wants_student else ["student", "graduate"])))
        if matched_fields and related_state:
            return STATUS_PARTIAL, ids_for(*(list(matched_fields) + ["student", "graduate"]))
        if matched_fields or related_state:
            return STATUS_PARTIAL, ids_for(*(list(matched_fields) + ["student", "graduate"]))
        return STATUS_NOT_EVIDENCED, []

    # Explicit CEFR threshold such as "at least C1 English". A lower level is
    # related evidence but must never be promoted to a full match.
    req_level = _best_cefr([requirement.text])
    if "english" in req and req_level:
        evidence_texts = [row["text"] for row in candidate_rows if "english" in _concepts_in_text(row["text"]) or _cefr_levels(row["text"])]
        ev_level = _best_cefr(evidence_texts)
        english_ids = ids_for("english")
        if ev_level and _CEFR_ORDER[ev_level] >= _CEFR_ORDER[req_level]:
            return STATUS_MATCHED, english_ids or _unique_ids([row["evidence_id"] for row in candidate_rows if _cefr_levels(row["text"])])
        if ev_level or "english" in present:
            ids = english_ids or _unique_ids([row["evidence_id"] for row in candidate_rows if _cefr_levels(row["text"])])
            return STATUS_PARTIAL, ids
        return STATUS_NOT_EVIDENCED, []

    # User/customer needs converted to business requirements. Direct user or
    # customer-needs evidence plus business-requirement work is a full match.
    if ({"user_needs", "customer_needs"} & req) and "business_needs" in req:
        direct_needs = {"user_needs", "customer_needs"} & present
        has_business = "business_needs" in present
        if direct_needs and has_business:
            return STATUS_MATCHED, ids_for(*(list(direct_needs) + ["business_needs"]))
        if direct_needs or has_business:
            return STATUS_PARTIAL, ids_for(*(list(direct_needs) + (["business_needs"] if has_business else [])))
        return STATUS_NOT_EVIDENCED, []

    # Explicit process-modeling + documentation requirement. The two parts may
    # be supported by different CV lines.
    if {"process_modeling", "documentation"}.issubset(req):
        found = {"process_modeling", "documentation"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("process_modeling", "documentation")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Simple explicit technical tools / technologies. Literal usage or listing is
    # sufficient when the requirement itself asks only for basic knowledge/use.
    for concept in (
        "python", "pandas", "csharp", "dotnet", "rest_api", "git", "aspnet_core",
        "entity_framework", "docker", "powerpoint", "process_modeling",
    ):
        if req == {concept}:
            return (STATUS_MATCHED, ids_for(concept)) if concept in present else (STATUS_NOT_EVIDENCED, [])

    # ASP.NET Core requirement may also trigger the generic .NET concept; the
    # explicit ASP.NET Core evidence is sufficient on its own.
    if "aspnet_core" in req:
        return (STATUS_MATCHED, ids_for("aspnet_core")) if "aspnet_core" in present else (STATUS_NOT_EVIDENCED, [])

    # C# + .NET is a compound explicit tool requirement.
    if {"csharp", "dotnet"}.issubset(req):
        found = {"csharp", "dotnet"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("csharp", "dotnet")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # PostgreSQL OR SQL Server: either named relational database satisfies the
    # alternative requirement.
    if "postgresql" in req or "sql_server" in req:
        found = {"postgresql", "sql_server"} & present
        if found:
            return STATUS_MATCHED, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # SQL + relational database knowledge can be supported by separate evidence.
    if {"sql", "relational_database"}.issubset(req):
        found = {"sql", "relational_database"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("sql", "relational_database")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Python data analysis and pandas usage are explicit activity requirements.
    if {"python", "data_analysis"}.issubset(req):
        found = {"python", "data_analysis"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("python", "data_analysis")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Excel used for analysis/reporting: all named parts can be evidenced across
    # multiple CV lines.
    if "excel" in req and ({"data_analysis", "reporting"} & req):
        needed = {"excel"} | ({"data_analysis", "reporting"} & req)
        found = needed & present
        if needed.issubset(present):
            return STATUS_MATCHED, ids_for(*needed)
        if "excel" in found and len(found) >= 2:
            return STATUS_PARTIAL, ids_for(*found)
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # General English requirement with no explicit CEFR threshold: an explicit
    # English level is enough.
    if "english" in req and not _best_cefr([requirement.text]) and not ({"written", "verbal", "communication"} & req):
        if "english" in present:
            return STATUS_MATCHED, ids_for("english")
        return STATUS_NOT_EVIDENCED, []

    # Machine-learning requirement: explicit ML tooling/model work is enough.
    if "machine_learning" in req:
        if "machine_learning" in present:
            return STATUS_MATCHED, ids_for("machine_learning")
        if "statistics_analysis" in present:
            return STATUS_PARTIAL, ids_for("statistics_analysis")
        return STATUS_NOT_EVIDENCED, []

    # Hypothesis testing / experiment analysis is stricter than generic
    # statistical analysis. Generic stats gives partial evidence only.
    if "hypothesis_testing" in req:
        if "hypothesis_testing" in present:
            return STATUS_MATCHED, ids_for("hypothesis_testing")
        if "statistics_analysis" in present:
            return STATUS_PARTIAL, ids_for("statistics_analysis")
        return STATUS_NOT_EVIDENCED, []

    # Analytical thinking + problem solving: explicit terms are strongest, but
    # substantive data-analysis/modeling work is legitimate partial evidence.
    if {"analytical", "problem_solving"}.issubset(req):
        found = {"analytical", "problem_solving"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("analytical", "problem_solving")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        if {"data_analysis", "statistics_analysis", "machine_learning"} & present:
            related = list({"data_analysis", "statistics_analysis", "machine_learning"} & present)
            return STATUS_PARTIAL, ids_for(*related)
        return STATUS_NOT_EVIDENCED, []

    # Communication + coordination, teamwork, planning/tracking and
    # reporting/documentation are common operations-role compound requirements.
    if {"communication", "coordination"}.issubset(req):
        found = {"communication", "coordination"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("communication", "coordination")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    if req == {"teamwork"}:
        return (STATUS_MATCHED, ids_for("teamwork")) if "teamwork" in present else (STATUS_NOT_EVIDENCED, [])

    if {"planning", "tracking"}.issubset(req):
        found = {"planning", "tracking"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("planning", "tracking")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    if {"reporting", "documentation"}.issubset(req):
        found = {"reporting", "documentation"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("reporting", "documentation")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Excel + PowerPoint basic-use requirement.
    if {"excel", "powerpoint"}.issubset(req):
        found = {"excel", "powerpoint"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("excel", "powerpoint")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Single Jira requirement is literal and should not be downgraded.
    if req == {"jira"}:
        return (STATUS_MATCHED, ids_for("jira")) if "jira" in present else (STATUS_NOT_EVIDENCED, [])

    # OOP is intentionally strict: using C# does not by itself prove OOP
    # knowledge unless the CV explicitly says so.
    if req == {"oop"}:
        return (STATUS_MATCHED, ids_for("oop")) if "oop" in present else (STATUS_NOT_EVIDENCED, [])

    # Basic project-management knowledge: explicit project-management wording is
    # full evidence; planning/coordination is only partial.
    if req == {"project_management"}:
        if "project_management" in present:
            return STATUS_MATCHED, ids_for("project_management")
        related = {"planning", "coordination", "tracking"} & present
        if related:
            return STATUS_PARTIAL, ids_for(*related)
        return STATUS_NOT_EVIDENCED, []

    # Compound interest: software projects + business analysis + project management.
    if {"business_analysis", "project_management"} & req and "software_development" in req:
        needed = {"software_development", "business_analysis", "project_management"} & req
        found = needed & present
        if needed and needed.issubset(present):
            return STATUS_MATCHED, ids_for(*needed)
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Customer/stakeholder needs + communication is intentionally strict.
    # Generic business-needs analysis is relevant, but it is not identical to
    # direct customer/stakeholder-needs evidence. Therefore:
    #   direct customer needs + communication -> matched
    #   business needs + communication (or only one side) -> partial
    if "customer_needs" in req and "communication" in req:
        has_direct_customer = "customer_needs" in present
        has_business_needs = "business_needs" in present
        has_communication = "communication" in present
        if has_direct_customer and has_communication:
            return STATUS_MATCHED, ids_for("customer_needs", "communication")
        if has_direct_customer or has_business_needs or has_communication:
            concepts: list[str] = []
            if has_direct_customer:
                concepts.append("customer_needs")
            elif has_business_needs:
                concepts.append("business_needs")
            if has_communication:
                concepts.append("communication")
            return STATUS_PARTIAL, ids_for(*concepts)
        return STATUS_NOT_EVIDENCED, []

    # Teamwork + communication: explicit team collaboration plus communication
    # evidence satisfies the compound soft-skill requirement.
    if {"teamwork", "communication"}.issubset(req):
        found = {"teamwork", "communication"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("teamwork", "communication")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Student-club / volunteer / project-organization responsibility.
    if "club_activity" in req or ("responsibility" in req and "student" in req):
        if "club_activity" in present and "responsibility" in present:
            return STATUS_MATCHED, ids_for("club_activity", "responsibility")
        if "club_activity" in present or "responsibility" in present:
            return STATUS_PARTIAL, ids_for(*(list({"club_activity", "responsibility"} & present)))
        return STATUS_NOT_EVIDENCED, []

    # Jira/Trello or similar named task-tracking tools. Generic task follow-up is
    # not enough to claim tool familiarity.
    if "jira" in req or "trello" in req:
        found_tools = {"jira", "trello"} & present
        if found_tools:
            return STATUS_MATCHED, ids_for(*found_tools)
        return STATUS_NOT_EVIDENCED, []

    # Teamwork + responsibility can be evidenced by explicit team/coordination and
    # role/task responsibility statements in the CV.
    if "teamwork" in req and "responsibility" in req:
        found = {"teamwork", "responsibility"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("teamwork", "responsibility")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Documentation + knowledge sharing is compound. Preparing SPEC/test documents
    # proves documentation, but not necessarily an explicit knowledge-sharing habit.
    if "documentation" in req and "knowledge_sharing" in req:
        found = {"documentation", "knowledge_sharing"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("documentation", "knowledge_sharing")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Learning/growth + solution motivation are soft traits and require explicit CV
    # wording; we intentionally do not infer them from unrelated experience.
    if "learning_growth" in req or "solution_motivation" in req:
        needed = {"learning_growth", "solution_motivation"} & req
        found = needed & present
        if needed and needed.issubset(present):
            return STATUS_MATCHED, ids_for(*needed)
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # A plain software-testing requirement is satisfied by explicit testing/UAT/test
    # planning evidence even if the CV does not literally say "software testing".
    if req == {"software_testing"} or ("software_testing" in req and "process_modeling" not in req):
        if "software_testing" in present:
            return STATUS_MATCHED, ids_for("software_testing")
        return STATUS_NOT_EVIDENCED, []

    # University-time software/project experience: explicit project work plus
    # development/tool evidence is enough; no model inference is required.
    if "project_experience" in req and "software_development" in req:
        found = {"project_experience", "software_development"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("project_experience", "software_development")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Single explicit technical-skill requirements. Literal CV evidence is
    # sufficient; a small model must not turn "SQL" into only a partial match.
    if req == {"sql"}:
        if "sql" in present:
            return STATUS_MATCHED, ids_for("sql")
        return STATUS_NOT_EVIDENCED, []

    # Explicit compound tool requirement: both SQL and Excel must be evidenced.
    if {"sql", "excel"}.issubset(req):
        found = {"sql", "excel"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("sql", "excel")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # Process modeling and software testing may be evidenced by different CV lines.
    if {"process_modeling", "software_testing"}.issubset(req):
        found = {"process_modeling", "software_testing"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("process_modeling", "software_testing")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    if {"analytical", "problem_solving"}.issubset(req):
        found = {"analytical", "problem_solving"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("analytical", "problem_solving")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    # "Written and verbal English communication" is intentionally stricter than
    # merely listing English proficiency. English alone is related evidence, not
    # proof of both written and verbal communication.
    if "english" in req and ({"written", "verbal", "communication"} & req):
        if "english" not in present:
            return STATUS_NOT_EVIDENCED, []
        communication_parts = {"written", "verbal", "communication"} & req
        explicit_parts = communication_parts & present
        if communication_parts and communication_parts.issubset(present):
            return STATUS_MATCHED, ids_for("english", *communication_parts)
        return STATUS_PARTIAL, ids_for("english", *explicit_parts)

    # Dashboard requirements are usually alternatives: Power BI OR another
    # dashboarding tool. Any explicit supported dashboard tool is enough.
    if "power_bi" in req or "dashboard" in req:
        if "power_bi" in present or "dashboard" in present:
            return STATUS_MATCHED, ids_for("power_bi", "dashboard")
        return STATUS_NOT_EVIDENCED, []

    # When both Agile and Jira are explicitly requested, require both for a full
    # match and one for a partial match.
    if {"agile", "jira"}.issubset(req):
        found = {"agile", "jira"} & present
        if len(found) == 2:
            return STATUS_MATCHED, ids_for("agile", "jira")
        if found:
            return STATUS_PARTIAL, ids_for(*found)
        return STATUS_NOT_EVIDENCED, []

    return None, []


def conservative_fallback_status(requirement: Requirement, candidate_rows: list[dict]) -> tuple[str, list[str]]:
    """Conservative fallback when the local model returns unusable JSON."""
    req_terms = _keyword_set(requirement.text)
    if not candidate_rows:
        return STATUS_NOT_EVIDENCED, []

    # First use the high-confidence career requirement rules when applicable.
    explicit_status, explicit_ids = explicit_requirement_status(requirement, candidate_rows)
    if explicit_status is not None:
        return explicit_status, explicit_ids

    scored_rows: list[tuple[float, float, dict]] = []
    for row in candidate_rows:
        ev_terms = _keyword_set(row["text"])
        overlap = len(req_terms & ev_terms) / max(len(req_terms), 1)
        sim = float(row.get("similarity", 0.0))
        lexical = float(row.get("lexical_score", 0.0))
        hybrid = 0.55 * overlap + 0.25 * max(0.0, min(sim, 1.0)) + 0.20 * lexical
        scored_rows.append((hybrid, max(overlap, lexical), row))

    _, overlap, best = max(scored_rows, key=lambda item: item[0])
    sim = float(best.get("similarity", 0.0))

    if overlap >= 0.42 and sim >= 0.68:
        return STATUS_MATCHED, [best["evidence_id"]]
    if overlap >= 0.20 or sim >= 0.82:
        return STATUS_PARTIAL, [best["evidence_id"]]
    return STATUS_NOT_EVIDENCED, []

def validate_classification(
    raw_model_output: str,
    requirements: list[Requirement],
    candidates: dict[str, list[dict]],
) -> list[dict]:
    """Validate model labels against literal, per-requirement CV evidence.

    The local model proposes a label, but deterministic checks have final authority
    for explicit career facts. This fixes false negatives such as an LLM saying
    "not evidenced" even though the CV literally lists both SQL and Excel.
    """

    parsed = _extract_json_array(raw_model_output) or []
    parsed_by_id: dict[str, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("requirement_id", ""))
        if rid and rid not in parsed_by_id:
            parsed_by_id[rid] = item

    guarded: list[dict] = []
    rank = {STATUS_NOT_EVIDENCED: 0, STATUS_PARTIAL: 1, STATUS_MATCHED: 2}

    for req in requirements:
        allowed_rows = candidates.get(req.id, [])
        allowed_ids = {row["evidence_id"] for row in allowed_rows}
        evidence_lookup = {row["evidence_id"]: row for row in allowed_rows}
        item = parsed_by_id.get(req.id)

        model_valid = False
        model_status = STATUS_NOT_EVIDENCED
        model_ids: list[str] = []
        if item:
            proposed_status = str(item.get("status", "")).strip().lower()
            proposed_ids = item.get("evidence_ids", [])
            if isinstance(proposed_ids, list):
                proposed_ids = [str(x) for x in proposed_ids]
                ids_valid = all(eid in allowed_ids for eid in proposed_ids)
                if proposed_status in ALLOWED_STATUSES and ids_valid:
                    if proposed_status == STATUS_NOT_EVIDENCED and not proposed_ids:
                        model_valid = True
                    elif proposed_status in {STATUS_MATCHED, STATUS_PARTIAL} and proposed_ids:
                        model_valid = True
                if model_valid:
                    model_status = proposed_status
                    model_ids = proposed_ids

        explicit_status, explicit_ids = explicit_requirement_status(req, allowed_rows)

        if explicit_status is not None:
            # Named/compound requirements are resolved deterministically from literal
            # CV evidence. This can safely correct a false-negative model label as
            # well as downgrade an unsupported overclaim.
            status = explicit_status
            evidence_ids = explicit_ids if explicit_status != STATUS_NOT_EVIDENCED else []
        elif model_valid:
            status = model_status
            evidence_ids = model_ids
            if status in {STATUS_MATCHED, STATUS_PARTIAL}:
                chosen_rows = [evidence_lookup[eid] for eid in evidence_ids if eid in evidence_lookup]
                guard_status, guard_ids = conservative_fallback_status(req, chosen_rows)
                # The deterministic guard may only keep or downgrade a generic model
                # claim; it never upgrades generic prose without an explicit rule.
                if rank[guard_status] < rank[status]:
                    status, evidence_ids = guard_status, guard_ids
        else:
            status, evidence_ids = conservative_fallback_status(req, allowed_rows)

        evidence = [evidence_lookup[eid] for eid in evidence_ids if eid in evidence_lookup]
        if status in {STATUS_MATCHED, STATUS_PARTIAL} and not evidence:
            status = STATUS_NOT_EVIDENCED
            evidence_ids = []
            evidence = []

        guarded.append(
            {
                "requirement_id": req.id,
                "requirement": req.text,
                "category": req.category,
                "status": status,
                "evidence": evidence,
                "note": _status_note(req, allowed_rows, status),
            }
        )

    return guarded

def _status_note(requirement: Requirement, candidate_rows: list[dict], status: str) -> str | None:
    req_level = _best_cefr([requirement.text])
    ev_level = _best_cefr([row["text"] for row in candidate_rows])
    if req_level and ev_level and status == STATUS_PARTIAL:
        return f"CV'de İngilizce seviyesi {ev_level}; ilan en az {req_level} istiyor."

    req = _concepts_in_text(requirement.text)
    present: set[str] = set()
    for row in candidate_rows:
        present |= _concepts_in_text(row["text"])
    if "graduate" in req and "student" not in req and "student" in present and "graduate" not in present and status == STATUS_PARTIAL:
        return "CV ilgili lisans alanında öğrencilik gösteriyor; ilan ise mezuniyet şartı istiyor."

    if "customer_needs" in req and "communication" in req and status == STATUS_PARTIAL:
        if "business_needs" in present and "customer_needs" not in present and "communication" in present:
            return (
                "CV iş ihtiyaçlarını analiz etme ve iletişim için ilişkili kanıt içeriyor; "
                "ancak doğrudan müşteri/stakeholder ihtiyacını analiz ettiğini açıkça göstermiyor."
            )

    if "documentation" in req and "knowledge_sharing" in req and status == STATUS_PARTIAL:
        if "documentation" in present and "knowledge_sharing" not in present:
            return (
                "CV doküman hazırlama konusunda açık kanıt içeriyor; "
                "bilgi paylaşımına ilişkin ayrı ve açık bir kanıt bulunmuyor."
            )
    return None


def summarize_guarded_results(items: list[dict]) -> dict:
    required = [x for x in items if x["category"] == "required"]
    preferred = [x for x in items if x["category"] == "preferred"]

    def counts(rows: list[dict]) -> dict[str, int]:
        return {
            STATUS_MATCHED: sum(1 for x in rows if x["status"] == STATUS_MATCHED),
            STATUS_PARTIAL: sum(1 for x in rows if x["status"] == STATUS_PARTIAL),
            STATUS_NOT_EVIDENCED: sum(1 for x in rows if x["status"] == STATUS_NOT_EVIDENCED),
            "total": len(rows),
        }

    required_counts = counts(required)
    preferred_counts = counts(preferred)

    if not required:
        fit = "Belirsiz"
        fit_note = "İş ilanında açık bir zorunlu gereksinim bölümü tespit edilemedi."
    else:
        weighted = (
            required_counts[STATUS_MATCHED]
            + 0.5 * required_counts[STATUS_PARTIAL]
        ) / max(required_counts["total"], 1)
        if required_counts[STATUS_NOT_EVIDENCED] == 0 and weighted >= 0.85:
            fit = "Güçlü uyum"
        elif weighted >= 0.60:
            fit = "İyi uyum"
        elif weighted >= 0.35:
            fit = "Kısmi uyum"
        else:
            fit = "Sınırlı uyum"
        fit_note = (
            "Bu sonuç yalnız yüklenen CV ve iş ilanındaki açık kanıtlara dayanır; "
            "işe alım olasılığı veya ATS puanı değildir."
        )

    return {
        "fit": fit,
        "fit_note": fit_note,
        "required_counts": required_counts,
        "preferred_counts": preferred_counts,
    }


def render_guarded_markdown(items: list[dict], summary: dict) -> str:
    status_label = {
        STATUS_MATCHED: "✅ Eşleşiyor",
        STATUS_PARTIAL: "🟡 Kısmen eşleşiyor",
        STATUS_NOT_EVIDENCED: "⚪ CV'de açık kanıt yok",
    }

    lines = [
        f"## Genel değerlendirme: {summary['fit']}",
        summary["fit_note"],
        "",
    ]

    rc = summary["required_counts"]
    if rc["total"]:
        lines.extend(
            [
                f"**Zorunlu gereksinimler:** {rc[STATUS_MATCHED]} eşleşiyor, "
                f"{rc[STATUS_PARTIAL]} kısmi, {rc[STATUS_NOT_EVIDENCED]} kanıt yok.",
                "",
                "### Gereksinim bazında değerlendirme",
            ]
        )

    def append_items(rows: list[dict]) -> None:
        for index, item in enumerate(rows, start=1):
            lines.append(f"**{index}. {status_label[item['status']]} — {item['requirement']}**")
            if item["evidence"]:
                for ev in item["evidence"]:
                    lines.append(f"- CV kanıtı ({ev['evidence_id']}): {ev['text']}")
            elif item["status"] == STATUS_NOT_EVIDENCED:
                lines.append("- Yüklenen CV'de bu gereksinimi açıkça doğrulayan kanıt bulunamadı.")
            if item.get("note"):
                lines.append(f"- Not: {item['note']}")
            elif item["status"] == STATUS_PARTIAL:
                lines.append("- Not: İlişkili kanıt var; ancak gereksinimin tamamı CV'de açıkça doğrulanmıyor.")
            lines.append("")

    append_items([x for x in items if x["category"] == "required"])

    preferred_rows = [x for x in items if x["category"] == "preferred"]
    if preferred_rows:
        lines.append("### Tercih edilen / artı nitelikler")
        append_items(preferred_rows)

    gaps = [x for x in items if x["status"] in {STATUS_PARTIAL, STATUS_NOT_EVIDENCED}]
    if gaps:
        lines.append("### CV'yi bu ilana göre iyileştirme")
        for item in gaps:
            if item["status"] == STATUS_NOT_EVIDENCED:
                lines.append(
                    f"- **{item['requirement']}**: Bu deneyim gerçekten varsa CV'de daha görünür ve somut biçimde belirt; yoksa ekleme."
                )
            else:
                lines.append(
                    f"- **{item['requirement']}**: Mevcut ilişkili deneyimi daha doğrudan anlat; yalnız gerçekten yaptığın kısmı yaz."
                )

    return "\n".join(lines).strip()
