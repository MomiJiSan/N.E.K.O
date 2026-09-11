"""Evaluate explicit local PCM16/16k/mono WAV fixtures through the real worker.

Manifest: JSON list of {id, expected: 'wake'|'none', wav?: path,
keyword_end_seconds?: number}. Without a labeled keyword endpoint, latency is
reported as input position at detection, not latency from completion of speech.
This tool never uploads or logs audio/transcripts. Use --synthetic to label TTS
fixtures honestly; synthetic measurements do not establish microphone recall.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import statistics
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.voice_wake_word import DEFAULT_WAKE_WORD_KEYWORDS
from main_logic.voice_input.activation.contracts import ActivationGeneration, AudioFrame
from main_logic.voice_input.wake_word.sherpa_backend import SherpaWakeWordConfig, SherpaWakeWordDetector


def read_fixture(path: Path) -> bytes:
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Fixture must be PCM16 mono 16 kHz")
        if audio.getnframes() > 16000 * 3600:
            raise ValueError("Fixture exceeds one-hour budget")
        return audio.readframes(audio.getnframes())


async def evaluate(args) -> dict:
    import psutil

    fixtures = json.loads(await asyncio.to_thread(args.manifest.read_text, encoding="utf-8-sig"))
    detector = SherpaWakeWordDetector(SherpaWakeWordConfig(
        str(args.model_dir), tuple(args.keyword or DEFAULT_WAKE_WORD_KEYWORDS),
        keyword_threshold=args.threshold))
    started = time.perf_counter()
    await detector.prepare()
    prepare_seconds = time.perf_counter() - started
    process = psutil.Process(detector._process.pid)
    peak_rss = 0
    results = []
    generation = ActivationGeneration("offline-evaluation", 1, 1, 1, 1, "fixture")
    try:
        for epoch, fixture in enumerate(fixtures):
            path = args.manifest.parent / fixture.get("wav", fixture["id"] + ".wav")
            pcm = await asyncio.to_thread(read_fixture, path)
            hits, feed_durations = [], []
            cpu_before = process.cpu_times()
            case_start = time.perf_counter()
            # Feed the WAV's actual silence. No synthetic padding is added.
            for offset in range(0, len(pcm), 640):
                chunk = pcm[offset:offset + 640]
                start = offset // 2
                audio = AudioFrame(offset // 640, start, start + len(chunk) // 2,
                    start / 16000, 16000, chunk, generation)
                before = time.perf_counter()
                result = await detector.feed(audio, epoch)
                feed_durations.append(time.perf_counter() - before)
                if result is not None:
                    hit = dict(keyword=result.keyword, sample_start=result.sample_start,
                               sample_end=result.sample_end, detected_at_input_seconds=audio.sample_end / 16000)
                    if "keyword_end_seconds" in fixture:
                        hit["input_latency_seconds"] = audio.sample_end / 16000 - fixture["keyword_end_seconds"]
                    hits.append(hit)
                if offset % 32000 == 0:
                    peak_rss = max(peak_rss, process.memory_info().rss)
            cpu_after = process.cpu_times()
            results.append(dict(id=fixture["id"], expected=fixture["expected"],
                duration_seconds=len(pcm) / 32000, hits=hits,
                wall_seconds=time.perf_counter() - case_start,
                worker_cpu_seconds=cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
                mean_feed_seconds=statistics.mean(feed_durations) if feed_durations else 0,
                max_feed_seconds=max(feed_durations, default=0)))
    finally:
        await detector.close()
    positive = [r for r in results if r["expected"] == "wake"]
    negative = [r for r in results if r["expected"] == "none"]
    negative_hours = sum(r["duration_seconds"] for r in negative) / 3600
    return dict(synthetic=args.synthetic, threshold=args.threshold,
        prepare_seconds=prepare_seconds, peak_worker_rss_bytes=peak_rss,
        positive_cases=len(positive), positive_hits=sum(bool(r["hits"]) for r in positive),
        negative_hours=negative_hours,
        false_hits=sum(len(r["hits"]) for r in negative),
        false_hits_per_hour=(sum(len(r["hits"]) for r in negative) / negative_hours if negative_hours else None),
        cases=results,
        limits="Offline PCM fixtures; not microphone/acoustic or downstream activation/transport acceptance.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--keyword", action="append", help="Override with encoded phonetic line; repeat per keyword")
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.output}; hits {report['positive_hits']}/{report['positive_cases']}; false hits {report['false_hits']}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
