"""Benchmark embedding performance across ONNX providers using real integration test data.

Tests against real PDF fixtures at various batch sizes.
"""

import asyncio
import os
import sys
import time

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_PDFS = {
    "nist-csf-2.0": os.path.join(FIXTURES_DIR, "nist-csf-2.0.pdf"),
    "nist-800-53r5": os.path.join(FIXTURES_DIR, "nist-800-53r5.pdf"),
}


async def bench_provider(provider_name: str, pdf_key: str):
    """Run the full index pipeline (load, chunk, embed) with a specific provider."""
    os.environ["ONNX_PROVIDER"] = provider_name

    from src.services.document_loader import load_document
    from src.services.chunker import chunk_document, ChunkOptions
    from src.services.embedder import Embedder

    pdf_path = FIXTURE_PDFS[pdf_key]
    embedder = Embedder()
    embedder._model = None
    embedder._providers = None

    # Load & chunk
    t0 = time.perf_counter()
    doc = await load_document(pdf_path)
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks, boundary_index = chunk_document(
        doc.content, pdf_key, ChunkOptions(),
        loader_boundaries=doc.boundaries or None,
    )
    chunk_time = time.perf_counter() - t0
    texts = [c.content for c in chunks]

    # Warmup (model load + provider init)
    t0 = time.perf_counter()
    _ = await embedder.embed_texts(texts[:5])
    warmup_time = time.perf_counter() - t0

    # Benchmark batch embedding at various batch sizes (3 runs each)
    batch_sizes_to_test = [8, 16, 32, 64, 128, 256]
    batch_results = {}
    for bs in batch_sizes_to_test:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            await embedder.embed_texts(texts, batch_size=bs)
            times.append(time.perf_counter() - t0)
        batch_results[bs] = times

    # Query embedding (5 runs)
    queries = [
        "What are the six functions of the NIST Cybersecurity Framework?",
        "How does the framework define implementation tiers?",
        "What is the purpose of the CSF organizational profile?",
        "How should organizations assess cybersecurity risk?",
        "What role does governance play in the framework?",
    ]
    query_times = []
    for q in queries:
        t0 = time.perf_counter()
        await embedder.embed_text(q)
        query_times.append(time.perf_counter() - t0)

    providers = embedder.active_providers
    batch_size = embedder.default_batch_size

    print(f"\n{'='*60}")
    print(f"Provider: {provider_name.upper()} | Doc: {pdf_key}")
    print(f"Active providers: {providers}")
    print(f"Default batch size: {batch_size}")
    print(f"{'='*60}")
    print(f"Chunks: {len(chunks)}")
    print(f"Load: {load_time*1000:.0f}ms | Chunk: {chunk_time*1000:.0f}ms | Warmup: {warmup_time*1000:.0f}ms")
    print(f"Batch embed {len(texts)} chunks (3 runs each):")
    best_bs = None
    best_mean = float("inf")
    for bs, times in sorted(batch_results.items()):
        mean = sum(times) / len(times) * 1000
        mn = min(times) * 1000
        tp = len(texts) / (sum(times) / len(times))
        if mean < best_mean:
            best_mean = mean
            best_bs = bs
        print(f"  batch_size={bs:>4}: mean={mean:7.1f}ms  min={mn:7.1f}ms  {tp:5.0f} chunks/sec")
    print(f"  >>> Best batch_size: {best_bs}")
    print(f"Query embed: mean={sum(query_times)/len(query_times)*1000:.1f}ms")

    default_times = batch_results[batch_size]
    return {
        "provider": provider_name,
        "pdf_key": pdf_key,
        "n_chunks": len(texts),
        "batch_results": {bs: sum(t)/len(t)*1000 for bs, t in batch_results.items()},
        "query_mean_ms": sum(query_times) / len(query_times) * 1000,
        "best_bs": best_bs,
        "best_mean_ms": best_mean,
    }


async def main():
    import onnxruntime as ort
    available = ort.get_available_providers()
    print(f"Available ONNX providers: {available}")

    pdf_key = sys.argv[1] if len(sys.argv) > 1 else "nist-csf-2.0"
    if pdf_key not in FIXTURE_PDFS:
        print(f"Unknown fixture: {pdf_key}. Choose from: {list(FIXTURE_PDFS.keys())}")
        sys.exit(1)
    if not os.path.exists(FIXTURE_PDFS[pdf_key]):
        print(f"Fixture not found: {FIXTURE_PDFS[pdf_key]}")
        sys.exit(1)

    providers_to_test = ["cpu"]
    if "CoreMLExecutionProvider" in available:
        providers_to_test.append("coreml")
    if "CUDAExecutionProvider" in available:
        providers_to_test.append("cuda")

    results = []
    for p in providers_to_test:
        r = await bench_provider(p, pdf_key)
        results.append(r)

    if len(results) > 1:
        baseline = results[0]
        print(f"\n{'='*60}")
        print(f"COMPARISON vs CPU ({baseline['n_chunks']} chunks from {pdf_key})")
        print(f"{'='*60}")
        for r in results[1:]:
            print(f"\n{r['provider'].upper()} vs CPU (per batch_size):")
            for bs in sorted(baseline["batch_results"]):
                cpu_ms = baseline["batch_results"][bs]
                accel_ms = r["batch_results"][bs]
                speedup = cpu_ms / accel_ms
                print(f"  batch_size={bs:>4}: CPU {cpu_ms:7.1f}ms  {r['provider'].upper()} {accel_ms:7.1f}ms  -> {speedup:.2f}x")
            query_speedup = baseline["query_mean_ms"] / r["query_mean_ms"]
            print(f"  Query embed: {query_speedup:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
