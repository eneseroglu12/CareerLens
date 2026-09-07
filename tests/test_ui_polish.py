from __future__ import annotations

from pathlib import Path
import unittest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class UIPolishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_text = APP_PATH.read_text(encoding="utf-8")

    def test_main_title_is_plain_brand_div_not_heading_anchor(self):
        self.assertIn('<div class="cl-hero-title">CV\'ni İş İlanlarıyla Karşılaştır</div>', self.app_text)
        self.assertNotIn('<h1>CV\'ni İş İlanlarıyla Karşılaştır</h1>', self.app_text)
        self.assertIn('[data-testid="stHeaderActionElements"]', self.app_text)

    def test_trust_indicators_are_next_to_main_title(self):
        title_pos = self.app_text.index('class="cl-hero-title"')
        trust_pos = self.app_text.index('class="cl-title-trust"')
        paragraph_pos = self.app_text.index("CV'ni ve iş ilanlarını yerel olarak analiz et")
        self.assertLess(title_pos, trust_pos)
        self.assertLess(trust_pos, paragraph_pos)
        self.assertIn("Yerel çalışır", self.app_text)
        self.assertIn("Belgeler cihazında kalır", self.app_text)
        self.assertIn("Kanıt denetimli", self.app_text)

    def test_chunk_jargon_is_removed_from_user_ui(self):
        for phrase in (
            'metric("Parça"',
            "metin parçası",
            "kanıt parçası",
            "parçalara ayrılır",
            "chunk_count",
        ):
            self.assertNotIn(phrase, self.app_text)
        self.assertIn("Toplam belge", self.app_text)

    def test_comparison_has_non_destructive_clear_button(self):
        self.assertIn('"Temizle"', self.app_text)
        self.assertIn("on_click=clear_match_workspace", self.app_text)
        self.assertIn("yüklenen belgeleri silmez", self.app_text)

    def test_document_reset_requires_confirmation(self):
        self.assertIn("Tüm belgeleri silmeyi onaylıyorum", self.app_text)
        self.assertIn("disabled=not confirm_reset", self.app_text)
        self.assertIn("on_click=reset_all_documents", self.app_text)

    def test_legacy_demo_documents_are_auto_cleaned(self):
        self.assertIn('LEGACY_DEMO_SOURCES = {"sample_cv.txt", "sample_job_description.txt", "career_guide.md"}', self.app_text)
        self.assertIn("cleanup_legacy_demo_sources", self.app_text)

    def test_raw_file_names_are_formatted_in_main_selectors(self):
        self.assertIn('format_func=lambda source: friendly_source_name(source, "cv")', self.app_text)
        self.assertIn('format_func=lambda source: friendly_source_name(source, "job")', self.app_text)

    def test_internal_classifier_method_is_not_shown_in_result_banner(self):
        self.assertNotIn("Değerlendirme yöntemi:", self.app_text)
        self.assertIn("CV–ilan eşleştirmesi", self.app_text)

    def test_sidebar_reopen_control_is_not_hidden(self):
        self.assertIn('[data-testid="stSidebarCollapsedControl"]', self.app_text)
        self.assertIn('visibility: visible !important', self.app_text)
        self.assertNotIn('#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }', self.app_text)

    def test_sidebar_starts_expanded_and_bootstraps_once(self):
        self.assertIn('initial_sidebar_state="expanded"', self.app_text)
        self.assertIn("ensure_sidebar_expanded_on_first_render", self.app_text)
        self.assertIn("_cl_sidebar_bootstrapped", self.app_text)
        self.assertIn('[data-testid="stSidebarCollapsedControl"]', self.app_text)

    def test_sidebar_uses_transparent_logo_asset(self):
        self.assertIn('kariyerlens_logo_transparent.png', self.app_text)
        self.assertIn('background:transparent;', self.app_text)
        self.assertIn('mix-blend-mode:normal;', self.app_text)

    def test_title_trust_labels_have_no_icon_badges(self):
        self.assertNotIn('class="cl-trust-icon"', self.app_text)
        self.assertNotIn('class="cl-trust-dot"', self.app_text)
        self.assertNotIn('<span class="cl-trust-icon">▣</span>', self.app_text)
        self.assertNotIn('<span class="cl-trust-icon">✓</span>', self.app_text)
        self.assertIn('<span class="cl-trust-sep">·</span>', self.app_text)

    def test_qa_ui_defaults_to_compact_evidence_count(self):
        self.assertIn('st.slider("Kullanılacak kanıt sayısı", 2, 5, 3)', self.app_text)
        self.assertIn('hızlı ve deterministik yanıt', self.app_text.lower())
        self.assertIn('Yerel modelle ayrıntılandır', self.app_text)
        self.assertIn('Analiz edilecek belge', self.app_text)


if __name__ == "__main__":
    unittest.main()
