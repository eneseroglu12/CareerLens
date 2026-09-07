from __future__ import annotations

from collections.abc import Callable
import gc

CHAT_MODEL_ALIAS = "qwen2.5-0.5b"
CHAT_MODEL_CPU_VARIANT = "qwen2.5-0.5b-instruct-generic-cpu:4"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
EMBEDDING_MODEL_CPU_VARIANT = "qwen3-embedding-0.6b-generic-cpu:1"

# Interactive Q&A favors latency on CPU. Retrieved context is deliberately
# compact, so a 2K window is sufficient and avoids a large KV-cache allocation.
CHAT_MAX_LENGTH = 2048
CHAT_FALLBACK_MAX_LENGTHS = (1536, 1024)

ProgressCallback = Callable[[str, float], None]


class FoundryRuntime:
    """Thin wrapper around the current Foundry Local Python SDK.

    Models are loaded sequentially to reduce memory pressure on laptops.
    """

    def __init__(self) -> None:
        self._manager = None
        self._eps_prepared = False
        self._chat_model = None

    def _get_manager(self):
        if self._manager is not None:
            return self._manager

        from foundry_local_sdk import Configuration, FoundryLocalManager

        try:
            FoundryLocalManager.initialize(Configuration(app_name="career_cv_foundry_rag"))
        except Exception as exc:
            # The SDK uses a process-wide singleton; repeated initialization can
            # raise in some versions. If an instance exists, reuse it.
            try:
                self._manager = FoundryLocalManager.instance
                return self._manager
            except Exception:
                raise RuntimeError(f"Foundry Local başlatılamadı: {exc}") from exc

        self._manager = FoundryLocalManager.instance
        return self._manager

    def prepare_execution_providers(self, progress_callback: ProgressCallback | None = None) -> None:
        if self._eps_prepared:
            return
        manager = self._get_manager()

        # Foundry Local invokes EP download progress from a native/CFFI callback.
        # Streamlit UI methods must not be called from that callback because a
        # rerun can raise StopException inside the CFFI boundary. Keep native
        # callbacks UI-free and report only coarse progress from this Python call.
        if progress_callback:
            progress_callback("Execution providers hazırlanıyor", 5.0)

        try:
            manager.download_and_register_eps()
        except Exception:
            # Generic CPU variants can still work even when an optional EP fails.
            pass

        self._eps_prepared = True
        if progress_callback:
            progress_callback("Execution providers hazır", 10.0)

    def _prepare_model(self, alias: str, progress_callback: ProgressCallback | None = None):
        manager = self._get_manager()

        # Use generic CPU variants for predictable behaviour across Windows systems.
        # Foundry Local still handles model download, loading and inference locally.
        if alias == CHAT_MODEL_ALIAS:
            preferred_cpu_variant = CHAT_MODEL_CPU_VARIANT
        elif alias == EMBEDDING_MODEL_ALIAS:
            # Keep the embedding model on the generic CPU variant as well.
            preferred_cpu_variant = EMBEDDING_MODEL_CPU_VARIANT
        else:
            preferred_cpu_variant = None

        if preferred_cpu_variant is not None:
            try:
                model = manager.catalog.get_model_variant(preferred_cpu_variant)
            except Exception:
                model = manager.catalog.get_model(alias)
                # Catalog revisions can change version suffixes. If the exact ID
                # is unavailable, select any generic CPU variant under the alias.
                try:
                    cpu_variant = next(
                        variant
                        for variant in getattr(model, "variants", [])
                        if "generic-cpu" in str(getattr(variant, "id", "")).lower()
                    )
                    model.select_variant(cpu_variant)
                except Exception:
                    # Last fallback keeps SDK behaviour and lets model.load()
                    # return a clear error if no CPU variant is actually usable.
                    pass
        else:
            model = manager.catalog.get_model(alias)

        # model.download(progress_callback=...) is also implemented through a
        # native callback. Do not forward Streamlit UI updates into that callback.
        if progress_callback:
            progress_callback(f"Model hazırlanıyor: {alias}", 12.0)
        model.download()
        model.load()
        if progress_callback:
            progress_callback(f"Model hazır: {alias}", 20.0)
        return model

    def _release_chat_model(self) -> None:
        """Free the cached chat model before a heavier embedding batch."""
        if self._chat_model is None:
            return
        try:
            self._chat_model.unload()
        except Exception:
            pass
        self._chat_model = None
        gc.collect()

    def _get_chat_model(self, progress_callback: ProgressCallback | None = None):
        """Load the compact chat model once and reuse it across questions."""
        if self._chat_model is None:
            self._chat_model = self._prepare_model(CHAT_MODEL_ALIAS, progress_callback)
        return self._chat_model

    def embed_texts(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
        batch_size: int = 8,
    ) -> list[list[float]]:
        if not texts:
            return []

        from foundry_local_sdk import EmbeddingsSession, Request, TensorItem, TextItem

        # Ingestion is infrequent. Free the cached chat model so low-memory
        # machines do not hold both model families at once.
        self._release_chat_model()
        self.prepare_execution_providers(progress_callback)
        model = self._prepare_model(EMBEDDING_MODEL_ALIAS, progress_callback)
        vectors: list[list[float]] = []

        try:
            with EmbeddingsSession(model) as session:
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    with Request() as request:
                        for text in batch:
                            request.add_item(TextItem(text))
                        with session.process_request(request) as response:
                            batch_vectors = [
                                list(item.data)
                                for item in response
                                if isinstance(item, TensorItem)
                            ]
                    if len(batch_vectors) != len(batch):
                        raise RuntimeError(
                            f"Embedding sayısı eşleşmedi: beklenen {len(batch)}, gelen {len(batch_vectors)}"
                        )
                    vectors.extend(batch_vectors)
                    if progress_callback:
                        done = min(start + len(batch), len(texts))
                        progress_callback("Embedding üretiliyor", 100.0 * done / len(texts))
        finally:
            try:
                model.unload()
            except Exception:
                pass
            # Foundry/ONNX releases some native buffers after Python references
            # are collected. Force a collection before loading the local chat
            # model to reduce Windows memory fragmentation in Streamlit.
            gc.collect()

        return vectors

    @staticmethod
    def _request_options(RequestOptions, SearchOptions, *, max_output_tokens: int, max_length: int):
        """Build Foundry options with an explicit ONNX GenAI max_length.

        Foundry Local SDK v2 exposes typed sampling parameters through
        SearchOptions and native passthrough parameters through
        RequestOptions.additional_options.  ``max_length`` is an ONNX GenAI
        generator search option and controls total sequence/KV-cache capacity.
        """
        search = SearchOptions(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )

        # Current SDK v2 API. Keep a compatibility fallback for minor SDK
        # revisions in which RequestOptions may expose the field after init.
        try:
            return RequestOptions(
                search=search,
                additional_options={"max_length": str(max_length)},
            )
        except TypeError:
            options = RequestOptions(search=search)
            if not hasattr(options, "additional_options"):
                raise RuntimeError(
                    "Yüklü Foundry Local SDK sürümü max_length passthrough seçeneğini "
                    "desteklemiyor. 'pip install --upgrade foundry-local-sdk' çalıştırın."
                )
            current = getattr(options, "additional_options", None)
            if current is None:
                try:
                    setattr(options, "additional_options", {})
                except Exception as exc:
                    raise RuntimeError(
                        "Foundry Local SDK üzerinde max_length ayarlanamadı."
                    ) from exc
                current = getattr(options, "additional_options")
            current["max_length"] = str(max_length)
            return options

    @staticmethod
    def _is_memory_allocation_error(exc: Exception) -> bool:
        """Recognize the allocation errors emitted by ONNX GenAI/Foundry.

        Depending on where allocation fails, the SDK may report a detailed
        key-value cache message or only ``failed to create generator: bad
        allocation``. Both mean we should retry with a smaller context window.
        """
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "key-value cache",
                "kv cache",
                "could not allocate",
                "bad allocation",
                "failed to create generator",
                "out of memory",
            )
        )

    def _run_chat_once(
        self,
        model,
        system_prompt: str,
        user_prompt: str,
        *,
        max_output_tokens: int,
        max_length: int,
    ) -> str:
        from foundry_local_sdk import (
            ChatSession,
            MessageItem,
            Request,
            RequestOptions,
            SearchOptions,
            TextItem,
        )

        parts: list[str] = []
        with ChatSession(model) as session:
            session.set_options(
                self._request_options(
                    RequestOptions,
                    SearchOptions,
                    max_output_tokens=max_output_tokens,
                    max_length=max_length,
                )
            )
            with Request() as request:
                request.add_item(MessageItem.system(system_prompt))
                request.add_item(MessageItem.user(user_prompt))
                with session.process_request(request) as response:
                    for item in response:
                        if isinstance(item, MessageItem):
                            for part in item.parts:
                                if isinstance(part, TextItem) and part.text:
                                    parts.append(part.text)
                        elif isinstance(item, TextItem) and item.text:
                            parts.append(item.text)

        answer = "".join(parts).strip()
        if not answer:
            raise RuntimeError("Yerel sohbet modeli boş cevap döndürdü.")
        return answer

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        progress_callback: ProgressCallback | None = None,
        max_output_tokens: int = 220,
    ) -> str:
        """Generate a compact local answer and keep the chat model warm.

        The model stays resident between questions and is released before a new
        embedding ingestion batch when memory is needed.
        """
        self.prepare_execution_providers(progress_callback)
        model = self._get_chat_model(progress_callback)

        lengths = (CHAT_MAX_LENGTH, *CHAT_FALLBACK_MAX_LENGTHS)
        last_exc: Exception | None = None
        for index, max_length in enumerate(lengths):
            try:
                if index and progress_callback:
                    progress_callback(
                        f"Bellek sınırı nedeniyle {max_length} token bağlamla yeniden deneniyor",
                        25.0 + index * 5.0,
                    )
                return self._run_chat_once(
                    model,
                    system_prompt,
                    user_prompt,
                    max_output_tokens=min(max_output_tokens, max(96, max_length // 5)),
                    max_length=max_length,
                )
            except Exception as exc:
                last_exc = exc
                if not self._is_memory_allocation_error(exc):
                    raise
                # A failed generator can leave native allocations fragmented.
                # Reload once with a smaller context window.
                self._release_chat_model()
                model = self._get_chat_model(progress_callback)
                gc.collect()
                continue
        assert last_exc is not None
        raise last_exc
