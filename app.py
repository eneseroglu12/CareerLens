from __future__ import annotations

import base64
import html
import re
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.database import (
    count_chunks,
    delete_source,
    initialize_database,
    list_sources,
    replace_source_chunks,
    reset_database,
    source_names,
)
from src.document_utils import extract_chunks
from src.foundry_runtime import (
    CHAT_MODEL_ALIAS,
    CHAT_MODEL_CPU_VARIANT,
    EMBEDDING_MODEL_ALIAS,
    EMBEDDING_MODEL_CPU_VARIANT,
)
from src.rag_engine import LocalRAGEngine
from src.query_router import route_question
from src.ui_evidence import redact_contacts, select_evidence_snippets

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "knowledge.db"
LOGO_PATH = APP_DIR / "assets" / "kariyerlens_logo_transparent.png"
LEGACY_DEMO_SOURCES = {"sample_cv.txt", "sample_job_description.txt", "career_guide.md"}

DOC_LABELS = {
    "cv": "CV / Özgeçmiş",
    "job": "İş İlanı",
    "guide": "Kariyer Rehberi",
}

STATUS_META = {
    "matched": {"label": "Eşleşiyor", "emoji": "✅", "css": "matched"},
    "partial": {"label": "Kısmen eşleşiyor", "emoji": "🟡", "css": "partial"},
    "not_evidenced": {"label": "CV'de açık kanıt yok", "emoji": "⚪", "css": "missing"},
}

st.set_page_config(
    page_title="CareerLens — Yerel Kariyer Asistanı",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _logo_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


LOGO_DATA_URI = _logo_data_uri(LOGO_PATH)

st.markdown(
    """
    <style>
    :root {
        --cl-bg-soft: #0f172a;
        --cl-panel: rgba(255, 255, 255, .03);
        --cl-panel-strong: rgba(255, 255, 255, .045);
        --cl-border: rgba(148, 163, 184, .16);
        --cl-muted: #94a3b8;
        --cl-muted-2: #cbd5e1;
        --cl-accent: #8b5cf6;
        --cl-accent-2: #6d28d9;
        --cl-green: #22c55e;
        --cl-yellow: #f59e0b;
        --cl-gray: #a1a1aa;
    }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        visibility: visible !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    .stAppDeployButton { display: none; }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 100000 !important;
    }
    [data-testid="stSidebarCollapsedControl"] button {
        border:1px solid rgba(148,163,184,.18) !important;
        background:rgba(15,23,42,.88) !important;
        border-radius:10px !important;
        backdrop-filter:blur(10px);
    }
    [data-testid="stHeaderActionElements"],
    .stHeading a,
    [data-testid="stMarkdownContainer"] h1 a,
    [data-testid="stMarkdownContainer"] h2 a,
    [data-testid="stMarkdownContainer"] h3 a,
    [data-testid="stMarkdownContainer"] h4 a { display:none !important; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1180px;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.35rem;
    }

    .cl-brand-wrap {
        display:flex;
        flex-direction:column;
        align-items:flex-start;
        margin:0 0 .15rem 0;
        padding:.15rem 0 .35rem 0;
    }
    .cl-brand-logo-shell {
        width:100%;
        overflow:visible;
        border-radius:0;
        background:transparent;
    }
    .cl-brand-logo-large {
        width:224px;
        max-width:100%;
        height:auto;
        display:block;
        margin:0;
        mix-blend-mode:normal;
        filter:drop-shadow(0 8px 20px rgba(0,0,0,.18));
        opacity:1;
    }
    .cl-brand-sub {
        color:var(--cl-muted);
        font-size:.82rem;
        margin:.3rem 0 1rem 0;
        line-height:1.45;
        letter-spacing:.005em;
    }
    .cl-sidebar-count {
        border:1px solid var(--cl-border);
        border-radius:15px;
        padding:.78rem .86rem;
        background:rgba(255,255,255,.018);
        margin:.15rem 0 .9rem 0;
    }
    .cl-sidebar-count-label {color:var(--cl-muted);font-size:.76rem;margin-bottom:.25rem;}
    .cl-sidebar-count-value {font-size:1.72rem;font-weight:760;line-height:1;}
    .cl-sidebar-count-sub {color:var(--cl-muted);font-size:.75rem;margin-top:.4rem;line-height:1.45;}
    .cl-source-row {
        padding:.22rem 0;
        color:#cbd5e1;
        font-size:.79rem;
        line-height:1.42;
        overflow-wrap:anywhere;
    }

    .cl-hero {
        padding:.25rem 0 1.2rem 0;
        margin-bottom:.35rem;
    }
    .cl-title-row {
        display:flex;
        align-items:center;
        justify-content:flex-start;
        gap:1rem;
        flex-wrap:wrap;
        margin-bottom:.62rem;
    }
    .cl-hero-title {
        font-size:2.45rem;
        line-height:1.06;
        letter-spacing:-.045em;
        margin:0;
        font-weight:790;
        color:#f8fafc;
    }
    .cl-title-trust {
        display:flex;
        align-items:center;
        gap:.48rem;
        flex-wrap:wrap;
        color:#9aa8ba;
        font-size:.76rem;
        white-space:nowrap;
        font-weight:520;
    }
    .cl-trust-item {display:inline-flex;align-items:center;opacity:.88;}
    .cl-trust-sep {color:#475569;opacity:.9;}
    .cl-hero p {
        margin:0;
        max-width:790px;
        color:var(--cl-muted);
        font-size:1.02rem;
        line-height:1.62;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color:var(--cl-border) !important;
        border-radius:18px !important;
        background:var(--cl-panel);
    }

    div[data-baseweb="tab-list"] {
        gap:.45rem;
        border-bottom:1px solid var(--cl-border);
        margin-bottom:1.5rem;
    }
    button[data-baseweb="tab"] {
        border-radius:10px 10px 0 0;
        padding:.58rem .82rem;
    }

    .stButton > button {
        border-radius:12px;
        font-weight:650;
        min-height:2.75rem;
        border:1px solid rgba(139,92,246,.15);
    }
    .stButton > button[kind="primary"] {
        box-shadow:0 8px 22px rgba(124,58,237,.18);
    }

    [data-testid="stFileUploaderDropzone"] {
        border:1px dashed rgba(148,163,184,.28);
        border-radius:14px;
        background:rgba(255,255,255,.018);
        padding:.68rem;
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-size:0 !important;
        color:transparent !important;
        position:relative;
    }
    [data-testid="stFileUploaderDropzone"] button span,
    [data-testid="stFileUploaderDropzone"] button p {font-size:0 !important;color:transparent !important;}
    [data-testid="stFileUploaderDropzone"] button svg {color:#e5e7eb !important;}
    [data-testid="stFileUploaderDropzone"] button::after {
        content:"Dosya seç";
        color:#e5e7eb !important;
        font-size:.86rem;
        font-weight:600;
        margin-left:.28rem;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span {font-size:0 !important;}
    [data-testid="stFileUploaderDropzoneInstructions"] span::after {
        content:"Dosyayı buraya sürükle veya bilgisayarından seç";
        font-size:.84rem;
        color:#cbd5e1;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] small {display:none !important;}

    [data-testid="stMetric"] {
        border:1px solid var(--cl-border);
        border-radius:15px;
        padding:.74rem .82rem;
        background:rgba(255,255,255,.018);
    }
    [data-testid="stMetricLabel"] {color:var(--cl-muted);}

    .cl-section-title {
        font-size:1.58rem;
        font-weight:755;
        letter-spacing:-.025em;
        margin-bottom:.25rem;
    }
    .cl-section-sub {
        color:var(--cl-muted);
        line-height:1.58;
        margin-bottom:1.22rem;
    }
    .cl-card-title {font-size:1.05rem;font-weight:700;margin-bottom:.15rem;}
    .cl-card-sub {font-size:.84rem;color:var(--cl-muted);margin-bottom:.72rem;}
    .cl-mini {font-size:.78rem;color:var(--cl-muted);line-height:1.56;}

    .cl-empty {
        border:1px dashed rgba(148,163,184,.24);
        border-radius:15px;
        padding:1.05rem 1.15rem;
        color:var(--cl-muted);
        background:rgba(255,255,255,.012);
    }

    .cl-summary-grid {
        display:grid;
        grid-template-columns:repeat(4, minmax(0, 1fr));
        gap:.85rem;
        margin:.35rem 0 1rem 0;
    }
    .cl-summary-card {
        border:1px solid var(--cl-border);
        border-radius:16px;
        background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
        padding:1rem 1rem .92rem 1rem;
    }
    .cl-summary-label {
        color:var(--cl-muted);
        font-size:.82rem;
        margin-bottom:.35rem;
    }
    .cl-summary-value {
        font-size:1.95rem;
        font-weight:760;
        line-height:1.05;
        letter-spacing:-.03em;
    }
    .cl-summary-sub {margin-top:.35rem;color:var(--cl-muted);font-size:.78rem;}

    .cl-fit-banner {
        border:1px solid var(--cl-border);
        border-radius:18px;
        padding:1rem 1.1rem;
        background:linear-gradient(135deg, rgba(139,92,246,.12), rgba(99,102,241,.08));
        margin-bottom:1rem;
    }
    .cl-fit-top {color:var(--cl-muted);font-size:.82rem;margin-bottom:.25rem;}
    .cl-fit-main {font-size:1.65rem;font-weight:790;letter-spacing:-.03em;line-height:1.08;}
    .cl-fit-note {margin-top:.45rem;color:#d7deea;line-height:1.55;font-size:.92rem;}

    .cl-group-title {
        font-size:1.08rem;
        font-weight:720;
        margin:1rem 0 .65rem 0;
        letter-spacing:-.02em;
    }

    .cl-req-card {
        border:1px solid var(--cl-border);
        border-radius:16px;
        background:rgba(255,255,255,.02);
        padding:.95rem 1rem;
        margin-bottom:.75rem;
    }
    .cl-req-head {
        display:flex;
        align-items:flex-start;
        gap:.6rem;
        margin-bottom:.55rem;
    }
    .cl-pill {
        display:inline-flex;
        align-items:center;
        gap:.35rem;
        padding:.22rem .58rem;
        border-radius:999px;
        font-size:.74rem;
        font-weight:700;
        border:1px solid var(--cl-border);
        white-space:nowrap;
        margin-top:.05rem;
    }
    .cl-pill.matched {background:rgba(34,197,94,.12); color:#b9f7c7;}
    .cl-pill.partial {background:rgba(245,158,11,.12); color:#f8d894;}
    .cl-pill.missing {background:rgba(161,161,170,.14); color:#e4e4e7;}
    .cl-req-title {
        font-size:1rem;
        font-weight:690;
        line-height:1.45;
        margin:0;
    }
    .cl-req-sub {
        color:var(--cl-muted);
        font-size:.79rem;
        margin-top:.16rem;
    }
    .cl-proof-box {
        margin-top:.45rem;
        border:1px solid rgba(148,163,184,.12);
        border-radius:12px;
        background:rgba(255,255,255,.02);
        padding:.72rem .78rem;
    }
    .cl-proof-title {font-size:.82rem;font-weight:680;margin-bottom:.42rem;color:#e5ecf6;}
    .cl-proof-item {
        font-size:.9rem;
        line-height:1.52;
        margin:.32rem 0;
        color:#d7deea;
    }
    .cl-proof-note {font-size:.84rem;color:var(--cl-muted);line-height:1.5;margin-top:.48rem;}

    .cl-improve-grid {
        display:grid;
        grid-template-columns:repeat(2, minmax(0, 1fr));
        gap:.8rem;
        margin-top:.35rem;
    }
    .cl-improve-card {
        border:1px solid var(--cl-border);
        border-radius:16px;
        background:rgba(255,255,255,.02);
        padding:.9rem .95rem;
    }
    .cl-improve-top {display:flex;align-items:center;gap:.45rem;margin-bottom:.45rem;}
    .cl-improve-tag {
        font-size:.72rem;
        font-weight:700;
        padding:.18rem .5rem;
        border-radius:999px;
        background:rgba(139,92,246,.14);
        color:#ddd6fe;
        border:1px solid rgba(139,92,246,.24);
    }
    .cl-improve-title {font-weight:690;line-height:1.45;font-size:.95rem;}
    .cl-improve-text {color:#d5deea;font-size:.88rem;line-height:1.55;}

    .cl-kv {
        border:1px solid var(--cl-border);
        border-radius:14px;
        padding:.9rem 1rem;
        background:rgba(255,255,255,.018);
        margin-top:.65rem;
    }

    .stExpander {
        border:1px solid var(--cl-border) !important;
        border-radius:14px !important;
        background:rgba(255,255,255,.02) !important;
    }

    hr {border-color:var(--cl-border) !important;}

    @media (max-width: 980px) {
        .cl-summary-grid, .cl-improve-grid {grid-template-columns:1fr 1fr;}
    }
    @media (max-width: 680px) {
        .cl-summary-grid, .cl-improve-grid {grid-template-columns:1fr;}
        .cl-req-head {flex-direction:column;}
        .cl-title-row {align-items:flex-start;}
        .cl-hero-title {font-size:2.05rem;}
        .cl-title-trust {gap:.58rem;white-space:normal;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_sidebar_expanded_on_first_render() -> None:
    """Open the sidebar once when a fresh Streamlit session starts.

    Streamlit can remember a previously collapsed sidebar in the browser even
    when ``initial_sidebar_state="expanded"`` is set. This tiny local component
    clicks Streamlit's own reopen control only on the first render. After that,
    the user remains free to collapse/reopen the sidebar normally.
    """
    if st.session_state.get("_cl_sidebar_bootstrapped"):
        return
    st.session_state["_cl_sidebar_bootstrapped"] = True
    components.html(
        """
        <script>
        (() => {
          const openSidebarIfNeeded = () => {
            try {
              const doc = window.parent.document;
              const sidebar = doc.querySelector('[data-testid="stSidebar"]');
              const reopen = doc.querySelector('[data-testid="stSidebarCollapsedControl"]');
              const collapsed = !sidebar || sidebar.getBoundingClientRect().width < 80;
              if (collapsed && reopen) {
                const button = reopen.querySelector('button') || reopen;
                button.click();
              }
            } catch (e) {
              // initial_sidebar_state="expanded" remains the safe fallback
            }
          };
          setTimeout(openSidebarIfNeeded, 120);
          setTimeout(openSidebarIfNeeded, 450);
          setTimeout(openSidebarIfNeeded, 900);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


ensure_sidebar_expanded_on_first_render()


@st.cache_resource(show_spinner=False)
def get_engine() -> LocalRAGEngine:
    return LocalRAGEngine()


initialize_database(DB_PATH)


def cleanup_legacy_demo_sources() -> list[str]:
    """Eski prototip sürümlerinden kalan örnek belgeleri sessizce temizle.

    Kullanıcının kendi test belgelerine dokunulmaz; yalnız CareerLens'in önceki
    sürümlerinde otomatik gelen üç sabit demo kaynak adı kaldırılır.
    """
    existing = {item["source"] for item in list_sources(DB_PATH)}
    removed: list[str] = []
    for source in sorted(LEGACY_DEMO_SOURCES & existing):
        delete_source(DB_PATH, source)
        removed.append(source)
    return removed


def friendly_source_name(source: str, doc_type: str | None = None) -> str:
    """Dosya adını kullanıcıya daha okunabilir bir etiket olarak göster.

    Veritabanında gerçek dosya adı korunur; yalnız arayüz etiketi temizlenir.
    """
    stem = Path(source).stem
    raw = stem
    raw = re.sub(r"^(?:cv|ilan|job)[_-]?\d*[_-]*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^(?:sample|demo)[_-]+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"[_-]+", " ", raw).strip()
    raw = re.sub(r"\btr\b$", "", raw, flags=re.IGNORECASE).strip()
    words = raw.split()

    # Test CV adlarında son iki kelime ad-soyad, önceki kelimeler rol olacak şekilde
    # daha doğal bir görünüm üret.
    if (doc_type == "cv" or stem.lower().startswith("cv_")) and len(words) >= 3:
        role = " ".join(words[:-2]).title()
        person = " ".join(words[-2:]).title()
        if role:
            return f"{person} — {role}"
        return person

    label = " ".join(words).title() if words else Path(source).name
    replacements = {
        "Cv": "CV", "Mis": "MIS", "Sql": "SQL", "Api": "API",
        "Ui": "UI", "Ux": "UX", "Bi": "BI", "Net": ".NET",
    }
    for old, new in replacements.items():
        label = re.sub(rf"\b{re.escape(old)}\b", new, label)
    return label or Path(source).name


def clear_match_workspace() -> None:
    for key in ("match_result", "match_result_pair", "match_cv_source", "match_job_source"):
        st.session_state.pop(key, None)


def clear_match_result_only() -> None:
    st.session_state.pop("match_result", None)
    st.session_state.pop("match_result_pair", None)


def reset_all_documents() -> None:
    reset_database(DB_PATH)
    clear_match_workspace()
    st.session_state.pop("delete_source_select", None)
    st.session_state.pop("confirm_reset_database", None)


_REMOVED_LEGACY_DEMOS = cleanup_legacy_demo_sources()


def ingest_uploads(uploaded_files, doc_type: str) -> tuple[int, int]:
    if not uploaded_files:
        return 0, 0

    parsed_records: list[tuple[str, list[dict]]] = []
    all_chunks: list[dict] = []
    for uploaded_file in uploaded_files:
        chunks = extract_chunks(uploaded_file.name, uploaded_file.getvalue(), doc_type)
        if chunks:
            parsed_records.append((uploaded_file.name, chunks))
            all_chunks.extend(chunks)

    if not all_chunks:
        return 0, 0

    durum = st.status("Belgeler hazırlanıyor…", expanded=True)
    ilerleme = st.progress(0)

    def on_progress(stage: str, percent: float) -> None:
        ilerleme.progress(
            max(0, min(int(percent), 100)),
            text=f"{stage}: %{percent:.0f}",
        )

    embeddings = get_engine().embed_texts(
        [item["content"] for item in all_chunks],
        progress_callback=on_progress,
    )

    cursor = 0
    for source_name, chunks in parsed_records:
        records = []
        for chunk in chunks:
            record = dict(chunk)
            record["embedding"] = embeddings[cursor]
            cursor += 1
            records.append(record)
        replace_source_chunks(DB_PATH, source_name, doc_type, records)

    ilerleme.progress(100, text="Yerel bilgi tabanına kaydedildi")
    durum.update(label="Belgeler hazır.", state="complete", expanded=False)
    return len(parsed_records), len(all_chunks)


def render_upload_card(title: str, subtitle: str, doc_type: str, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"<div class='cl-card-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='cl-card-sub'>{subtitle}</div>", unsafe_allow_html=True)
        files = st.file_uploader(
            f"{title} dosyaları",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key=f"{key}_upload",
            label_visibility="collapsed",
        )
        st.caption("PDF, DOCX, TXT veya MD")
        if st.button(
            "Yükle ve hazırla",
            type="primary",
            use_container_width=True,
            key=f"{key}_index",
            disabled=not files,
        ):
            try:
                docs, chunks = ingest_uploads(files, doc_type)
                if docs:
                    st.success(f"{docs} belge hazırlandı.")
                    st.rerun()
                else:
                    st.warning("Dosyada okunabilir metin bulunamadı.")
            except Exception as exc:
                st.error("Belge hazırlanırken bir hata oluştu.")
                st.exception(exc)


def improvement_text(status: str, requirement_text: str) -> str:
    if status == "not_evidenced":
        return (
            "Bu deneyim gerçekten varsa CV'de bunu daha görünür, kısa ve somut bir cümleyle belirt. "
            "Yoksa ekleme yapma."
        )
    return (
        "İlgili deneyim zaten var görünüyor; bunu bu gereksinime daha doğrudan bağlayan daha net bir ifade kullan. "
        "Yalnız gerçekten yaptığın kısmı yaz."
    )


def render_status_pill(status: str) -> str:
    meta = STATUS_META[status]
    return (
        f"<span class='cl-pill {meta['css']}'>{meta['emoji']} {html.escape(meta['label'])}</span>"
    )


def render_match_results(result: dict) -> None:
    summary = result["summary"]
    counts = summary["required_counts"]
    preferred_counts = summary.get("preferred_counts", {})
    guarded_items = result.get("guarded_items", [])
    required_items = [item for item in guarded_items if item.get("category") == "required"]
    preferred_items = [item for item in guarded_items if item.get("category") == "preferred"]
    improvement_items = [
        item for item in guarded_items if item.get("status") in {"partial", "not_evidenced"}
    ]

    st.markdown("### Sonuç")

    st.markdown(
        f"""
        <div class="cl-fit-banner">
          <div class="cl-fit-top">Genel değerlendirme</div>
          <div class="cl-fit-main">{html.escape(summary['fit'])}</div>
          <div class="cl-fit-note">{html.escape(summary['fit_note'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="cl-summary-grid">
          <div class="cl-summary-card">
            <div class="cl-summary-label">Zorunlu gereksinim</div>
            <div class="cl-summary-value">{counts['total']}</div>
            <div class="cl-summary-sub">Toplam değerlendirilen madde</div>
          </div>
          <div class="cl-summary-card">
            <div class="cl-summary-label">Eşleşen</div>
            <div class="cl-summary-value">{counts['matched']}</div>
            <div class="cl-summary-sub">Açık kanıtla desteklenenler</div>
          </div>
          <div class="cl-summary-card">
            <div class="cl-summary-label">Kısmi</div>
            <div class="cl-summary-value">{counts['partial']}</div>
            <div class="cl-summary-sub">İlişkili ama eksik destek</div>
          </div>
          <div class="cl-summary-card">
            <div class="cl-summary-label">Kanıt yok</div>
            <div class="cl-summary-value">{counts['not_evidenced']}</div>
            <div class="cl-summary-sub">CV'de açıkça görünmeyenler</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Kısa özet")
    st.markdown(
        f"- **Zorunlu gereksinimler:** {counts['matched']} eşleşiyor, {counts['partial']} kısmi, {counts['not_evidenced']} kanıt yok."
    )
    if preferred_counts.get("total"):
        st.markdown(
            f"- **Tercih edilen nitelikler:** {preferred_counts.get('matched', 0)} eşleşiyor, {preferred_counts.get('partial', 0)} kısmi, {preferred_counts.get('not_evidenced', 0)} kanıt yok."
        )

    with st.expander("Ayrıntılı kanıtları aç", expanded=False):
        st.markdown("<div class='cl-group-title'>Zorunlu gereksinimler</div>", unsafe_allow_html=True)
        for index, item in enumerate(required_items, start=1):
            proof_lines = select_evidence_snippets(item["requirement"], item.get("evidence", []), limit=2)
            header_html = (
                f"<div class='cl-req-card'>"
                f"<div class='cl-req-head'>"
                f"{render_status_pill(item['status'])}"
                f"<div>"
                f"<div class='cl-req-title'>{index}. {html.escape(item['requirement'])}</div>"
                f"<div class='cl-req-sub'>{'Zorunlu gereksinim'}</div>"
                f"</div></div>"
            )
            if item["status"] == "not_evidenced":
                body_html = (
                    "<div class='cl-proof-box'><div class='cl-proof-title'>Kanıt özeti</div>"
                    "<div class='cl-proof-item'>Yüklenen CV'de bu gereksinimi açıkça doğrulayan kısa ve temiz bir kanıt cümlesi bulunamadı.</div>"
                    "</div>"
                )
            else:
                lines_html = "".join(
                    f"<div class='cl-proof-item'><strong>{html.escape(eid)}</strong> · {html.escape(text)}</div>"
                    for eid, text in proof_lines
                )
                if not lines_html:
                    lines_html = "<div class='cl-proof-item'>Uygun kanıt bulundu; ancak gösterim için temiz kısa cümle üretilemedi.</div>"
                note_html = ""
                if item.get("note"):
                    note_html = f"<div class='cl-proof-note'>Not: {html.escape(item['note'])}</div>"
                elif item["status"] == "partial":
                    note_html = (
                        "<div class='cl-proof-note'>Not: İlişkili kanıt var; ancak gereksinimin tamamı CV'de açıkça doğrulanmıyor.</div>"
                    )
                body_html = (
                    f"<div class='cl-proof-box'><div class='cl-proof-title'>Kanıt özeti</div>{lines_html}{note_html}</div>"
                )
            st.markdown(header_html + body_html + "</div>", unsafe_allow_html=True)

        if preferred_items:
            st.markdown("<div class='cl-group-title'>Tercih edilen / artı nitelikler</div>", unsafe_allow_html=True)
            for index, item in enumerate(preferred_items, start=1):
                proof_lines = select_evidence_snippets(item["requirement"], item.get("evidence", []), limit=2)
                header_html = (
                    f"<div class='cl-req-card'>"
                    f"<div class='cl-req-head'>"
                    f"{render_status_pill(item['status'])}"
                    f"<div>"
                    f"<div class='cl-req-title'>{index}. {html.escape(item['requirement'])}</div>"
                    f"<div class='cl-req-sub'>{'Tercih edilen nitelik'}</div>"
                    f"</div></div>"
                )
                if item["status"] == "not_evidenced":
                    body_html = (
                        "<div class='cl-proof-box'><div class='cl-proof-title'>Kanıt özeti</div>"
                        "<div class='cl-proof-item'>Yüklenen CV'de bu niteliği açıkça doğrulayan kısa ve temiz bir kanıt cümlesi bulunamadı.</div>"
                        "</div>"
                    )
                else:
                    lines_html = "".join(
                        f"<div class='cl-proof-item'><strong>{html.escape(eid)}</strong> · {html.escape(text)}</div>"
                        for eid, text in proof_lines
                    )
                    if not lines_html:
                        lines_html = "<div class='cl-proof-item'>Uygun kanıt bulundu; ancak gösterim için temiz kısa cümle üretilemedi.</div>"
                    note_html = ""
                    if item.get("note"):
                        note_html = f"<div class='cl-proof-note'>Not: {html.escape(item['note'])}</div>"
                    elif item["status"] == "partial":
                        note_html = (
                            "<div class='cl-proof-note'>Not: İlişkili kanıt var; ancak gereksinimin tamamı CV'de açıkça doğrulanmıyor.</div>"
                        )
                    body_html = (
                        f"<div class='cl-proof-box'><div class='cl-proof-title'>Kanıt özeti</div>{lines_html}{note_html}</div>"
                    )
                st.markdown(header_html + body_html + "</div>", unsafe_allow_html=True)

    if improvement_items:
        st.markdown("### CV'yi iyileştirme önerileri")
        st.markdown(
            "İstersen önce aşağıdaki boşlukları kapatacak kısa, somut ve doğrulanabilir cümleler ekleyebilirsin.")
        cards_html = ["<div class='cl-improve-grid'>"]
        for item in improvement_items:
            tag = "Eksik kanıt" if item["status"] == "not_evidenced" else "Kısmi kanıt"
            cards_html.append(
                f"<div class='cl-improve-card'>"
                f"<div class='cl-improve-top'><span class='cl-improve-tag'>{html.escape(tag)}</span></div>"
                f"<div class='cl-improve-title'>{html.escape(item['requirement'])}</div>"
                f"<div class='cl-improve-text'>{html.escape(improvement_text(item['status'], item['requirement']))}</div>"
                f"</div>"
            )
        cards_html.append("</div>")
        st.markdown("".join(cards_html), unsafe_allow_html=True)

    overlap = result["overlap"]
    with st.expander("Anahtar kelime örtüşmesini göster", expanded=False):
        st.markdown(
            f"<div class='cl-kv'><strong>Kelime örtüşmesi:</strong> %{overlap['coverage'] * 100:.0f}<br><span class='cl-mini'>Bu gösterge yalnız kelime örtüşmesini ölçer; ATS puanı değildir.</span></div>",
            unsafe_allow_html=True,
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**CV'de geçen terimler**")
            st.write(", ".join(overlap.get("matched", [])) or "—")
        with cc2:
            st.markdown("**Aynı ifadeyle görünmeyenler**")
            st.write(", ".join(overlap.get("missing", [])) or "—")


# ── Sol panel ────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_DATA_URI:
        st.markdown(
            f"""
            <div class="cl-brand-wrap">
              <div class="cl-brand-logo-shell">
                <img class="cl-brand-logo-large" src="{LOGO_DATA_URI}" alt="CareerLens logosu" />
              </div>
            </div>
            <div class="cl-brand-sub">Yerel kariyer çalışma alanın</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="cl-brand-wrap" style="font-size:1.45rem;font-weight:760;color:white">CareerLens</div>
            <div class="cl-brand-sub">Yerel kariyer çalışma alanın</div>
            """,
            unsafe_allow_html=True,
        )

    sources = list_sources(DB_PATH)
    total_docs = len(sources)
    cv_count = sum(1 for item in sources if item["doc_type"] == "cv")
    job_count = sum(1 for item in sources if item["doc_type"] == "job")
    guide_count = sum(1 for item in sources if item["doc_type"] == "guide")

    count_sub = f"{cv_count} CV · {job_count} iş ilanı"
    if guide_count:
        count_sub += f" · {guide_count} ek kaynak"
    st.markdown(
        f"""
        <div class="cl-sidebar-count">
          <div class="cl-sidebar-count-label">Toplam belge</div>
          <div class="cl-sidebar-count-value">{total_docs}</div>
          <div class="cl-sidebar-count-sub">{html.escape(count_sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sources:
        st.markdown("#### Belgeler")
        groups = [("cv", "CV'ler"), ("job", "İş ilanları"), ("guide", "Ek kaynaklar")]
        for doc_type, heading in groups:
            items = [item for item in sources if item["doc_type"] == doc_type]
            if items:
                st.caption(heading)
                for item in items:
                    label = friendly_source_name(item["source"], doc_type)
                    st.markdown(
                        f"<div class='cl-source-row'>• {html.escape(label)}</div>",
                        unsafe_allow_html=True,
                    )

        with st.expander("Belgeleri yönet"):
            source_type_map = {item["source"]: item["doc_type"] for item in sources}
            selected_delete = st.selectbox(
                "Silinecek belge",
                [item["source"] for item in sources],
                format_func=lambda source: friendly_source_name(source, source_type_map.get(source)),
                key="delete_source_select",
            )
            if st.button("Seçili belgeyi sil", use_container_width=True):
                delete_source(DB_PATH, selected_delete)
                clear_match_workspace()
                st.rerun()

            st.markdown("<div class='cl-mini' style='margin-top:.65rem'>Tüm belgeleri temizlemek geri alınamaz.</div>", unsafe_allow_html=True)
            confirm_reset = st.checkbox("Tüm belgeleri silmeyi onaylıyorum", key="confirm_reset_database")
            st.button(
                "Tüm belgeleri temizle",
                use_container_width=True,
                disabled=not confirm_reset,
                on_click=reset_all_documents,
                key="reset_database_button",
            )
    else:
        st.markdown(
            "<div class='cl-empty'>Henüz belge yok.<br>İlk CV'ni ve iş ilanını ekleyerek başla.</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Teknik bilgiler"):
        st.caption("Sohbet modeli")
        st.code(CHAT_MODEL_ALIAS, language="text")
        st.caption("Anlamsal vektör modeli")
        st.code(EMBEDDING_MODEL_ALIAS, language="text")
        st.caption("CPU çalışma varyantları")
        st.code(f"{CHAT_MODEL_CPU_VARIANT}\n{EMBEDDING_MODEL_CPU_VARIANT}", language="text")
        st.caption("CV–ilan eşleştirmesi")
        st.write("Gereksinim bazlı kanıt motoru ve deterministik Python doğrulaması kullanılır.")

    with st.expander("Nasıl çalışır?"):
        st.markdown(
            """
            1. Belgeler cihazında okunur ve anlamlı analiz bölümlerine ayrılır.  
            2. Foundry Local bu bölümleri anlamsal vektörlere dönüştürür.  
            3. Sorularda hızlı yerel metin aramasıyla en ilgili kanıtlar seçilir; belge hazırlığında Foundry Local vektörleri de saklanır.  
            4. CV–ilan eşleştirmesinde her gereksinim ayrıca deterministik doğrulamadan geçer.  
            5. Sonuç ekranında yalnız kısa ve ilgili kanıt cümleleri gösterilir.
            """
        )

    st.markdown(
        "<div class='cl-mini' style='margin-top:1rem'>Belgeler ve model çıkarımları bu cihazdaki yerel çalışma alanında kalır.</div>",
        unsafe_allow_html=True,
    )


# ── Üst alan ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="cl-hero">
      <div class="cl-title-row">
        <div class="cl-hero-title">CV'ni İş İlanlarıyla Karşılaştır</div>
        <div class="cl-title-trust" aria-label="Yerel çalışma özellikleri">
          <span class="cl-trust-item">Yerel çalışır</span>
          <span class="cl-trust-sep">·</span>
          <span class="cl-trust-item">Belgeler cihazında kalır</span>
          <span class="cl-trust-sep">·</span>
          <span class="cl-trust-item">Kanıt denetimli</span>
        </div>
      </div>
      <p>CV'ni ve iş ilanlarını yerel olarak analiz et. CareerLens sonuçları yalnız yüklediğin belgelere dayanır; sonuç ekranı kısa tutulur, istersen ayrıntılı kanıtları sonradan açabilirsin.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

belgeler_tab, sor_tab, eslestir_tab = st.tabs(
    ["1 · Belgeler", "2 · CareerLens'e Sor", "3 · CV ↔ İş İlanı"]
)


# ── Belgeler ────────────────────────────────────────────────────────────────
with belgeler_tab:
    st.markdown("<div class='cl-section-title'>Belgelerini ekle</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cl-section-sub'>Başlamak için bir CV ve bir iş ilanı yükle. İstersen ek kariyer rehberleri de ekleyebilirsin. Dosyalar yalnız bu cihazdaki yerel bilgi tabanına hazırlanır.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        render_upload_card(
            "CV / Özgeçmiş",
            "Eğitim, deneyim, proje ve becerilerini içeren dosyan.",
            "cv",
            "cv",
        )
    with right:
        render_upload_card(
            "İş İlanı",
            "Karşılaştırmak istediğin pozisyonun ilan metni.",
            "job",
            "job",
        )

    with st.expander("Kariyer rehberi veya ek kaynak ekle — isteğe bağlı"):
        st.caption(
            "Genel kariyer sorularında kullanılmasını istediğin rehber, not veya kaynakları buraya ekleyebilirsin."
        )
        guide_files = st.file_uploader(
            "Kariyer rehberi dosyaları",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="guide_upload",
            label_visibility="collapsed",
        )
        if st.button(
            "Ek kaynakları yükle ve hazırla",
            use_container_width=True,
            key="guide_index",
            disabled=not guide_files,
        ):
            try:
                docs, chunks = ingest_uploads(guide_files, "guide")
                if docs:
                    st.success(f"{docs} ek kaynak hazırlandı.")
                    st.rerun()
                else:
                    st.warning("Dosyada okunabilir metin bulunamadı.")
            except Exception as exc:
                st.error("Belge hazırlanırken bir hata oluştu.")
                st.exception(exc)


# ── Soru-cevap ──────────────────────────────────────────────────────────────
with sor_tab:
    st.markdown("<div class='cl-section-title'>Belgelerine soru sor</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cl-section-sub'>CareerLens sorunun hangi belge türüyle ilgili olduğunu belirler ve yalnız ilgili kanıtlardan hızlı bir yanıt oluşturur. İstersen arama ayarlarından yerel modelle ayrıntılandırabilirsin.</div>",
        unsafe_allow_html=True,
    )

    if count_chunks(DB_PATH) == 0:
        st.info("Önce Belgeler sekmesinden en az bir belge eklemelisin.")
    else:
        with st.expander("Arama ayarları"):
            scope = st.selectbox(
                "Arama kapsamı",
                [
                    "Otomatik — önerilen",
                    "Tüm belgeler",
                    "Yalnız CV",
                    "Yalnız iş ilanları",
                    "Yalnız kariyer rehberleri",
                ],
            )
            top_k = st.slider("Kullanılacak kanıt sayısı", 2, 5, 3)
            detailed_local_model = st.checkbox(
                "Yerel modelle ayrıntılandır",
                value=False,
                help="Kapalıyken CareerLens kanıtlardan hızlı ve deterministik yanıt üretir. Açıkken yerel dil modeli de çalışır ve belirgin biçimde daha yavaş olabilir.",
            )

        scope_map = {
            "Otomatik — önerilen": None,
            "Tüm belgeler": [],
            "Yalnız CV": ["cv"],
            "Yalnız iş ilanları": ["job"],
            "Yalnız kariyer rehberleri": ["guide"],
        }

        question = st.text_area(
            "Sorun",
            placeholder=(
                "Örnek: CV'mde hangi teknik beceriler öne çıkıyor?\n"
                "Örnek: Bu iş ilanının zorunlu gereksinimleri neler?"
            ),
            height=125,
        )

        qa_sources = None
        if question.strip():
            preview_route = route_question(question.strip()) if scope == "Otomatik — önerilen" else None
            manual_types = scope_map[scope]
            candidate_types = preview_route.get("doc_types", []) if preview_route else (manual_types or [])
            if len(candidate_types) == 1:
                candidate_type = candidate_types[0]
                candidates = source_names(DB_PATH, candidate_type)
                if len(candidates) > 1:
                    selected_qa_source = st.selectbox(
                        "Analiz edilecek belge",
                        candidates,
                        format_func=lambda source: friendly_source_name(source, candidate_type),
                        help="Aynı türde birden fazla belge yüklü olduğu için CareerLens'in hangi belgeyi kullanacağını seç.",
                        key=f"qa_source_{candidate_type}",
                    )
                    qa_sources = [selected_qa_source]

        if st.button(
            "Yanıtı hazırla",
            type="primary",
            disabled=not question.strip(),
            use_container_width=False,
        ):
            try:
                started = time.perf_counter()
                spinner_text = (
                    "İlgili kanıtlar seçiliyor ve yerel model yanıt hazırlıyor…"
                    if detailed_local_model
                    else "İlgili kanıtlar seçiliyor…"
                )
                with st.spinner(spinner_text):
                    answer, results, route = get_engine().answer(
                        question=question.strip(),
                        db_path=DB_PATH,
                        top_k=top_k,
                        doc_types=scope_map[scope],
                        sources=qa_sources,
                        use_model=detailed_local_model,
                    )
                elapsed = time.perf_counter() - started

                mode_label = route.get("answer_mode", "Hızlı kanıt yanıtı")
                st.caption(
                    f"Kullanılan kapsam: **{route['label']}** · {mode_label} · **{elapsed:.1f} sn**"
                )
                st.caption(route['reason'])
                if route.get("selected_cv") and route.get("selected_job"):
                    st.caption(
                        "Karşılaştırılan belgeler: "
                        f"**{friendly_source_name(route['selected_cv'], 'cv')}** ↔ "
                        f"**{friendly_source_name(route['selected_job'], 'job')}**"
                    )
                elif route.get("selected_source"):
                    selected_type = route.get("doc_types", [None])[0] if len(route.get("doc_types", [])) == 1 else None
                    st.caption(
                        "Analiz edilen belge: "
                        f"**{friendly_source_name(route['selected_source'], selected_type)}**"
                    )

                st.markdown("### Yanıt")
                with st.container(border=True):
                    st.markdown(answer)

                with st.expander(f"Kullanılan kanıtları göster · {len(results)} kanıt"):
                    for index, item in enumerate(results, start=1):
                        page = f" · sayfa {item['page']}" if item.get("page") else ""
                        with st.container(border=True):
                            source_label = friendly_source_name(item["source"], item.get("doc_type"))
                            st.markdown(
                                f"**{index}. {source_label}** · "
                                f"{DOC_LABELS.get(item['doc_type'], item['doc_type'])}{page}"
                            )
                            st.caption(f"İlgililik: {item['score']:.2f}")
                            sanitized = redact_contacts(item["content"])
                            if len(sanitized) > 450:
                                sanitized = sanitized[:447].rstrip() + "…"
                            st.write(sanitized)
            except Exception as exc:
                st.error("Yanıt hazırlanırken bir hata oluştu.")
                st.exception(exc)


# ── CV ↔ İş ilanı ───────────────────────────────────────────────────────────
with eslestir_tab:
    st.markdown("<div class='cl-section-title'>CV'ni iş ilanıyla karşılaştır</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cl-section-sub'>İlandaki gereksinimler tek tek çıkarılır ve her madde yalnız CV'ndeki açık kanıtlarla değerlendirilir.</div>",
        unsafe_allow_html=True,
    )

    cvs = source_names(DB_PATH, "cv")
    jobs = source_names(DB_PATH, "job")

    if not cvs or not jobs:
        st.markdown(
            "<div class='cl-empty'>Eşleştirme için en az bir CV ve bir iş ilanı gerekiyor. Belgeler sekmesinden ikisini de ekleyebilirsin.</div>",
            unsafe_allow_html=True,
        )
    else:
        # Veri havuzu değiştiyse artık geçerli olmayan seçimleri temizle.
        if st.session_state.get("match_cv_source") not in cvs:
            st.session_state.pop("match_cv_source", None)
        if st.session_state.get("match_job_source") not in jobs:
            st.session_state.pop("match_job_source", None)

        with st.container(border=True):
            col1, col2 = st.columns(2, gap="large")
            cv_source = col1.selectbox(
                "CV",
                cvs,
                key="match_cv_source",
                format_func=lambda source: friendly_source_name(source, "cv"),
            )
            job_source = col2.selectbox(
                "İş ilanı",
                jobs,
                key="match_job_source",
                format_func=lambda source: friendly_source_name(source, "job"),
            )
            st.caption(
                "✅ Eşleşiyor · 🟡 Kısmen eşleşiyor · ⚪ CV'de açık kanıt yok. "
                "Kanıt bulunmaması, adayın o beceriye sahip olmadığı anlamına gelmez."
            )
            action_col, clear_col = st.columns([4, 1])
            analyze = action_col.button(
                "Eşleşmeyi analiz et",
                type="primary",
                use_container_width=True,
                key="analyze_match_button",
            )
            clear_col.button(
                "Temizle",
                use_container_width=True,
                key="clear_match_button",
                help="Seçimi ve ekrandaki karşılaştırma sonucunu sıfırlar; yüklenen belgeleri silmez.",
                on_click=clear_match_workspace,
            )

        if analyze:
            try:
                with st.spinner("Gereksinimler çıkarılıyor ve CV kanıtları denetleniyor…"):
                    result = get_engine().analyze_match(
                        db_path=DB_PATH,
                        cv_source=cv_source,
                        job_source=job_source,
                    )
                st.session_state["match_result"] = result
                st.session_state["match_result_pair"] = (cv_source, job_source)
            except Exception as exc:
                clear_match_result_only()
                st.error("Eşleştirme analizi hazırlanırken bir hata oluştu.")
                st.exception(exc)

        saved_result = st.session_state.get("match_result")
        saved_pair = st.session_state.get("match_result_pair")
        if saved_result and saved_pair == (cv_source, job_source):
            render_match_results(saved_result)
