from __future__ import annotations

import sys
import traceback

from src.foundry_runtime import (
    CHAT_MODEL_ALIAS,
    CHAT_MODEL_CPU_VARIANT,
    CHAT_MAX_LENGTH,
    CHAT_FALLBACK_MAX_LENGTHS,
    EMBEDDING_MODEL_ALIAS,
    EMBEDDING_MODEL_CPU_VARIANT,
    FoundryRuntime,
)


def main() -> int:
    runtime = FoundryRuntime()
    print("Foundry Local çalışma zamanı testi başlıyor...")
    print(f"Embedding modeli: {EMBEDDING_MODEL_ALIAS}")
    print(f"Sabitlenen embedding CPU varyantı: {EMBEDDING_MODEL_CPU_VARIANT}")
    vector = runtime.embed_texts(["Business analyst with SQL and process analysis experience."])[0]
    print(f"Embedding OK. Boyut: {len(vector)}")

    print(f"Sohbet modeli: {CHAT_MODEL_ALIAS}")
    print(f"Sabitlenen CPU varyantı: {CHAT_MODEL_CPU_VARIANT}")
    print(f"Bellek dostu sohbet bağlamı: {CHAT_MAX_LENGTH} token (fallback zinciri: {CHAT_FALLBACK_MAX_LENGTHS})")
    answer = runtime.chat(
        "Yalnız verilen doğrulama bilgisini kullan. Ek bilgi ekleme. Çok kısa cevap ver.",
        "DOĞRULAMA BİLGİSİ: Kod 4821.\nSORU: Kod nedir?",
        max_output_tokens=40,
    )
    print("Sohbet OK:", answer)
    print("TEST BAŞARILI: Foundry Local embedding ve hızlı 0.5B sohbet modeli çalışıyor.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("\nTEST BAŞARISIZ")
        traceback.print_exc()
        sys.exit(1)
