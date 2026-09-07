from __future__ import annotations

import unittest

from src.ui_evidence import redact_contacts, sentence_candidates, select_evidence_snippets


class UIEvidenceTests(unittest.TestCase):
    def test_contact_details_are_removed(self):
        text = "Enes • +90 555 111 2233 • enes@example.com • linkedin.com/in/enes • github.com/enes SQL"
        cleaned = redact_contacts(text)
        self.assertNotIn("555", cleaned)
        self.assertNotIn("@", cleaned)
        self.assertNotIn("linkedin", cleaned.lower())
        self.assertNotIn("github", cleaned.lower())

    def test_flattened_skills_and_language_are_split(self):
        text = (
            "Teknolojiler: Flutter, Firebase, Dart BECERILER Teknik Beceriler: Python, C#, SQL, Microsoft Excel "
            "Yetkinlikler: İş Analizi, Gereksinim Analizi, Fonksiyonel Analiz, Test Planlama "
            "Diller: İngilizce – B2 (Üst-Orta Seviye) 2022 – 2027 (Beklenen) Yönetim Kurulu Üyesi"
        )
        parts = sentence_candidates(text)
        self.assertIn("Diller: İngilizce – B2 (Üst-Orta Seviye)", parts)
        self.assertTrue(any("SQL" in p for p in parts))

    def test_english_requirement_shows_only_language_evidence(self):
        rows = [{
            "evidence_id": "E8",
            "text": (
                "Teknolojiler: Flutter, Firebase, Dart BECERILER Teknik Beceriler: Python, C#, SQL, Microsoft Excel "
                "Yetkinlikler: İş Analizi, Gereksinim Analizi, Fonksiyonel Analiz, Test Planlama "
                "Diller: İngilizce – B2 (Üst-Orta Seviye)"
            ),
        }]
        result = select_evidence_snippets("En az C1 düzeyinde İngilizce bilgisine sahip olmak.", rows)
        self.assertEqual(len(result), 1)
        self.assertIn("İngilizce", result[0][1])
        self.assertNotIn("Flutter", result[0][1])

    def test_sql_requirement_uses_one_concise_skill_line(self):
        rows = [{
            "evidence_id": "E8",
            "text": "Teknik Beceriler: Python, C#, SQL, Microsoft Excel Yetkinlikler: İş Analizi, Test Planlama",
        }, {
            "evidence_id": "E3",
            "text": "Uygulama ekranları, iş gereksinimleri, veri tabanı tabloları ve PL/SQL prosedürlerini inceleyerek iş ve sistem analizi çalışmalarında görev aldım.",
        }]
        result = select_evidence_snippets("Temel düzeyde SQL bilgisine sahip olmak.", rows)
        self.assertEqual(len(result), 1)
        self.assertIn("SQL", result[0][1])

    def test_testing_prefers_concrete_action_evidence(self):
        rows = [{
            "evidence_id": "E8",
            "text": "Yetkinlikler: İş Analizi, Gereksinim Analizi, Fonksiyonel Analiz, Test Planlama",
        }, {
            "evidence_id": "E4",
            "text": "İş gereksinimleri doğrultusunda fonksiyonel analiz ve SPEC dokümanları, test planları ve test senaryoları hazırladım.",
        }]
        result = select_evidence_snippets("Yazılım test süreçleri hakkında temel bilgi sahibi olmak.", rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "E4")
        self.assertIn("test senaryoları", result[0][1])

    def test_second_evidence_only_when_it_adds_another_requirement_aspect(self):
        rows = [{"evidence_id": "E2", "text": "Teknik altyapımı iş ihtiyaçlarını analiz etmek için kullanıyorum."},
                {"evidence_id": "E6", "text": "Kulüp çalışmalarında ekip içi iletişim ve koordinasyonda görev aldım."},
                {"evidence_id": "E7", "text": "Python ve SQL ile projeler geliştirdim."}]
        result = select_evidence_snippets(
            "Müşteri ihtiyaçlarını anlamaya istekli olmak ve etkili iletişim kurabilmek.", rows
        )
        self.assertEqual(len(result), 2)
        text = " ".join(item[1].lower() for item in result)
        self.assertIn("ihtiyaç", text)
        self.assertIn("iletişim", text)
        self.assertNotIn("python", text)

    def test_no_more_than_two_evidence_snippets(self):
        rows = [
            {"evidence_id": "E1", "text": "SQL kullandım."},
            {"evidence_id": "E2", "text": "SQL sorguları hazırladım."},
            {"evidence_id": "E3", "text": "SQL ile veri kontrolü yaptım."},
        ]
        result = select_evidence_snippets("SQL bilgisine sahip olmak.", rows, limit=2)
        self.assertLessEqual(len(result), 2)

    def test_turkish_class_ordinal_is_not_split_into_fragment(self):
        from src.requirement_guard import extract_cv_evidence_units
        cv = (
            "Yönetim Bilişim Sistemleri 4. sınıf öğrencisiyim. "
            "İş analizi, veri ve yazılım teknolojileriyle ilgileniyorum; Python ve SQL alanlarında teknik bilgiye sahibim. "
            "Teknik altyapımı iş ihtiyaçlarını analiz etmek ve teknoloji odaklı çözümler geliştirmek için kullanıyorum. "
            "Bu ek cümle satırı 240 karakterin üzerine taşımak için eklenmiştir."
        )
        texts = [u.text for u in extract_cv_evidence_units(cv)]
        self.assertTrue(any(text.startswith("Yönetim Bilişim Sistemleri 4. sınıf öğrencisiyim") for text in texts))
        self.assertNotIn("Yönetim Bilişim Sistemleri 4.", texts)

    def test_degree_line_is_retained_as_cv_evidence(self):
        from src.requirement_guard import extract_cv_evidence_units
        units = extract_cv_evidence_units("EĞİTİM\nYönetim Bilişim Sistemleri (Tam Burslu) – Lisans")
        self.assertTrue(any("Yönetim Bilişim Sistemleri" in u.text and "Lisans" in u.text for u in units))

    def test_realistic_turkish_cv_job_integration_is_stable(self):
        from src.rag_engine import LocalRAGEngine
        cv = """Yönetim Bilişim Sistemleri 4. sınıf öğrencisiyim. İş analizi, veri ve yazılım teknolojileriyle ilgileniyorum; Python, C# ve SQL alanlarında teknik bilgiye sahibim. Teknik altyapımı iş ihtiyaçlarını analiz etmek ve teknoloji odaklı çözümler geliştirmek için kullanıyorum.
DENEYİM
Uygulama ekranları, iş gereksinimleri, veri tabanı tabloları ve PL/SQL prosedürlerini inceleyerek iş ve sistem analizi çalışmalarında görev aldım.
İş gereksinimleri doğrultusunda fonksiyonel analiz ve SPEC dokümanları, test planları ve test senaryoları hazırladım.
EĞİTİM
Yönetim Bilişim Sistemleri (Tam Burslu) – Lisans
PROJELER
Üniversite Kampüs Uygulaması
Öğrenci ve kulüplere yönelik kampüs uygulamasında etkinlik yönetimi, kullanıcı rolleri ve uygulama içi iletişim özelliklerini geliştirdim.
Teknolojiler: Flutter, Firebase, Dart
Teknik Beceriler: Python, C#, SQL, Microsoft Excel
Yetkinlikler: İş Analizi, Gereksinim Analizi, Fonksiyonel Analiz, Test Planlama
Diller: İngilizce – B2 (Üst-Orta Seviye)
KULÜP AKTİVİTELERİ
Kulüp faaliyetleri ve etkinliklerin planlanması, organizasyonu ve koordinasyonunda görev alarak ekip içi iletişim, görev dağılımı ve yönetim kurulu karar süreçlerine katkı sağladım."""
        job = """ARANAN NİTELİKLER
- Mühendislik, Bilgisayar Bilimleri, Bilgi Teknolojileri, Yönetim Bilişim Sistemleri veya benzeri bir lisans programından mezun olmak.
- Yazılım projeleri, iş analizi ve proje yönetimine ilgi duymak.
- Müşteri ihtiyaçlarını anlamaya istekli olmak ve etkili iletişim kurabilmek.
- Takım çalışmasına yatkın olmak ve sorumluluk almaya istekli olmak.
- Dokümantasyona ve bilgi paylaşımına önem vermek.
- En az C1 düzeyinde İngilizce bilgisine sahip olmak.
- Öğrenmeye açık, gelişim odaklı ve çözüm üretmeye motive olmak.
TERCİH EDİLEN NİTELİKLER
- Yazılım test süreçleri hakkında temel bilgi sahibi olmak.
- Temel düzeyde SQL bilgisine sahip olmak.
- Üniversite döneminde yazılım geliştirme veya proje bazlı çalışmalara katılmış olmak."""
        result = LocalRAGEngine()._guarded_match_from_texts(cv, job)
        self.assertEqual(result["summary"]["fit"], "Kısmi uyum")
        self.assertEqual(result["summary"]["required_counts"], {"matched": 1, "partial": 5, "not_evidenced": 1, "total": 7})
        self.assertEqual(result["summary"]["preferred_counts"], {"matched": 3, "partial": 0, "not_evidenced": 0, "total": 3})
        english = next(x for x in result["guarded_items"] if "C1" in x["requirement"])
        english_snippets = select_evidence_snippets(english["requirement"], english["evidence"])
        self.assertEqual(len(english_snippets), 1)
        self.assertIn("B2", english_snippets[0][1])


if __name__ == "__main__":
    unittest.main()
