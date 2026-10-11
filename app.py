#!/usr/bin/env python3
"""
YouTube Fact Checker — PySide6 Application
=================================================
All features: Batch queue, subtitle fallback, per-task LLM,
faster-whisper, Wikipedia grounding, credibility scoring,
claim splitting, manual curation, session save/load,
CSV/PDF export, contradiction matrix.
"""

import sys, os, json, re, time, gc, logging, subprocess, csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus, urlparse

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLineEdit, QTextEdit, QLabel,
    QComboBox, QProgressBar, QFileDialog, QGroupBox, QCheckBox,
    QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QMessageBox, QAbstractItemView, QGridLayout,
    QDialog, QDialogButtonBox, QPlainTextEdit, QRadioButton,
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings, QTimer
from PySide6.QtGui import (
    QFont, QColor, QPalette, QTextCursor, QTextCharFormat, QBrush,
)
from PySide6.QtPrintSupport import QPrinter

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FactChecker")

APP_NAME = "YouTube Fact Checker"
APP_VERSION = "0.0.0"
DEFAULT_OUT_DIR = str(Path.home() / "fact_checker_output")
DEFAULT_OLLAMA = "http://localhost:11434"
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
RECOMMENDED_LLM = "llama3.2"
FAST_LLM = "phi3"
LLM_MODELS = [
    "llama3.2", "llama3.1", "llama3", "mistral", "gemma2",
    "qwen2.5", "phi3", "phi3.5", "command-r",
]

VERDICT_COLORS = {
    "True": QColor(39, 174, 96), "False": QColor(231, 76, 60),
    "Partially True": QColor(243, 156, 18),
    "Unverified": QColor(149, 165, 166),
    "Misleading": QColor(230, 126, 34),
}
SPEAKER_COLORS = [
    QColor(41, 128, 185), QColor(39, 174, 96), QColor(192, 57, 43),
    QColor(142, 68, 173), QColor(243, 156, 18), QColor(22, 160, 133),
    QColor(211, 84, 0), QColor(44, 62, 80), QColor(127, 140, 141),
]

TIER1_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
    "nature.com", "who.int", "cdc.gov", "nih.gov", "nasa.gov",
    "wikipedia.org", "britannica.com",
}
TIER2_DOMAINS = {
    "theguardian.com", "npr.org", "cnn.com", "wsj.com",
    "snopes.com", "politifact.com", "factcheck.org",
}
TIER_UNRELIABLE = {
    "infowars.com", "naturalnews.com", "breitbart.com",
    "rt.com", "theonion.com", "babylonbee.com",
}


# ───────────────────────── Helpers ─────────────────────────

def fmt_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_ts(ts):
    ts = ts.replace(',', '.')
    parts = ts.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0


def parse_vtt(text):
    entries = []
    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*'
        r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*\n((?:.*\n)*?)(?=\n\n|\Z)',
        re.MULTILINE)
    for m in pattern.finditer(text):
        txt = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        txt = re.sub(r'\n', ' ', txt)
        if txt:
            entries.append({
                "start": _parse_ts(m.group(1)),
                "end": _parse_ts(m.group(2)),
                "text": txt,
                "speaker": "Speaker 1",
                "words": [],
            })
    return entries


def parse_srt(text):
    entries = []
    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*'
        r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*\n((?:.*\n)*?)(?=\n\n|\Z)',
        re.MULTILINE)
    for m in pattern.finditer(text):
        txt = m.group(3).strip()
        txt = re.sub(r'\n', ' ', txt)
        if txt:
            entries.append({
                "start": _parse_ts(m.group(1)),
                "end": _parse_ts(m.group(2)),
                "text": txt,
                "speaker": "Speaker 1",
                "words": [],
            })
    return entries


def safe_filename(text, max_len=40):
    """Sanitize text for use as a filename — no backslash in f-string."""
    clean = re.sub(r'[^\w\s-]', '', text)
    return clean[:max_len]


def ollama_request(url, model, prompt, system="", stream=False):
    ep = f"{url.rstrip('/')}/api/generate"
    payload = {
        "model": model, "prompt": prompt, "system": system,
        "stream": stream,
        "options": {"temperature": 0.3, "num_predict": 4096},
    }
    try:
        r = requests.post(ep, json=payload, timeout=300, stream=stream)
        r.raise_for_status()
        if stream:
            full = ""
            for line in r.iter_lines():
                if line:
                    d = json.loads(line)
                    if "response" in d:
                        full += d["response"]
                    if d.get("done"):
                        break
            return full
        return r.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama failed: {e}")
        return ""


def ollama_list_models(url):
    try:
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_pull_model(url, name, cb=None):
    ep = f"{url.rstrip('/')}/api/pull"
    r = requests.post(ep, json={"name": name, "stream": True},
                      timeout=600, stream=True)
    r.raise_for_status()
    for line in r.iter_lines():
        if line:
            d = json.loads(line)
            if cb:
                cb(d.get("status", ""))
            if d.get("error"):
                raise Exception(d["error"])


def get_optimal_whisper_device_compute():
    """
    Determines the optimal device and compute type for faster-whisper.
    - GTX 1080 (Pascal, sm_61): 'cuda' + 'int8' (float16 is not 'efficient' here)
    - RTX 2000+ (Turing+ sm_75+): 'cuda' + 'float16' (Tensor cores available)
    - No GPU: 'cpu' + 'int8'
    """
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 7:
                # Volta, Turing, Ampere, Ada: Have tensor cores for efficient float16
                return "cuda", "float16"
            else:
                # Pascal (GTX 10-series like 1080), Maxwell, etc.
                # float16 is NOT efficiently supported, but int8 on CUDA is fast
                return "cuda", "int8"
    except ImportError:
        pass
    
    # Fallback if torch isn't available or no CUDA
    try:
        import ctranslate2
        if "cuda" in ctranslate2.get_supported_devices():
            return "cuda", "int8"
    except ImportError:
        pass
        
    return "cpu", "int8"

# ───────────────────── Wikipedia API ─────────────────────

def wikipedia_search(query, max_results=3):
    results = []
    # Wikipedia requires a descriptive User-Agent header
    headers = {
        "User-Agent": f"{APP_NAME}/{APP_VERSION} (https://github.com/yourproject; your_email@example.com)"
    }
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": max_results,
                "format": "json", "utf8": 1,
            }, headers=headers, timeout=10) # Added headers
        r.raise_for_status()
        for sr in r.json().get("query", {}).get("search", []):
            title = sr.get("title", "")
            if not title:
                continue
            try:
                slug = quote_plus(title.replace(' ', '_'))
                sresp = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
                    headers=headers, timeout=8) # Added headers
                if sresp.status_code != 200:
                    continue
                sdata = sresp.json()
                if sdata.get("type") == "disambiguation":
                    continue
                body = sdata.get("extract", "")
                if not body or len(body) < 30:
                    continue
                href = (sdata.get("content_urls", {})
                        .get("desktop", {})
                        .get("page", f"https://en.wikipedia.org/wiki/{slug}"))
                results.append({
                    "title": title, "body": body, "href": href,
                    "source": "wikipedia", "credibility": 85,
                })
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
    return results


def wikipedia_extract_entities(claim, ollama_url="", llm_model=""):
    named = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', claim)
    if ollama_url and llm_model and not named:
        try:
            resp = ollama_request(
                ollama_url, llm_model,
                f'Claim: "{claim}"',
                "Extract main entities. Return ONLY JSON array of strings.")
            if resp:
                m = re.search(r'\[[\s\S]*?\]', resp)
                if m:
                    named.extend(json.loads(m.group()))
        except Exception:
            pass
    seen = set()
    return [e for e in named
            if e.lower() not in seen and not seen.add(e.lower()) and len(e) > 2][:5]


# ───────────────── Source Credibility ─────────────────

def score_source_credibility(source_url, title="", body="",
                             ollama_url="", llm_model=""):
    try:
        domain = re.sub(r'^www\.', '', urlparse(source_url).netloc.lower())
    except Exception:
        domain = source_url.lower()

    factors = []
    score = 50
    tier = "unknown"

    for t1 in TIER1_DOMAINS:
        if t1 in domain or domain.endswith(t1):
            score = 92; tier = "tier1"
            factors.append(f"High credibility domain ({domain})")
            break
    if tier == "unknown":
        for t2 in TIER2_DOMAINS:
            if t2 in domain:
                score = 75; tier = "tier2"
                factors.append(f"Established outlet ({domain})")
                break
    if tier == "unknown":
        for tu in TIER_UNRELIABLE:
            if tu in domain:
                score = 20; tier = "unreliable"
                factors.append(f"Flagged unreliable ({domain})")
                break
    if tier == "unknown":
        if domain.endswith(".gov"):
            score = 95; tier = "tier1"
            factors.append("Official government")
        elif domain.endswith(".edu"):
            score = 85; tier = "tier1"
            factors.append("Academic institution")
        elif domain.endswith(".org"):
            score = 65; tier = "tier2"
            factors.append("Organizational source")

    if body:
        indicators = ["according to", "study", "research", "data",
                      "published", "university"]
        matches = sum(1 for ind in indicators if ind in body.lower())
        if matches >= 2:
            score = min(score + 10, 100)

    return {
        "score": max(0, min(100, score)),
        "tier": tier, "domain": domain, "factors": factors,
    }


def compute_weighted_verdict(results_with_cred):
    if not results_with_cred:
        return {"weighted_score": 0, "dominant_tier": "unknown"}
    tier_weights = {"tier1": 0, "tier2": 0, "tier3": 0,
                    "unreliable": 0, "unknown": 0}
    for item in results_with_cred:
        t = item.get("tier", "unknown")
        tier_weights[t] = tier_weights.get(t, 0) + item.get("credibility", 50)
    avg = sum(r.get("credibility", 50) for r in results_with_cred) / len(results_with_cred)
    credible = sum(1 for r in results_with_cred if r.get("credibility", 0) >= 60)
    unreliable = sum(1 for r in results_with_cred if r.get("credibility", 0) < 40)
    return {
        "weighted_score": round(avg, 1),
        "dominant_tier": max(tier_weights, key=tier_weights.get),
        "credible_count": credible,
        "unreliable_count": unreliable,
        "total_sources": len(results_with_cred),
    }


# ───────────────── Web Search ─────────────────

def web_search(query, max_results=5):
    try:
        # BUG FIX: Handle list input for query
        if isinstance(query, list):
            query = " ".join(str(x) for x in query)
        if not isinstance(query, str):
            query = str(query)

        from ddgs import DDGS
            
        out = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                    "source": "duckduckgo",
                })
        return out
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

def wikipedia_search(query, max_results=3):
    results = []
    headers = {
        "User-Agent": f"{APP_NAME}/{APP_VERSION} (https://github.com/yourproject; your_email@example.com)"
    }
    try:
        if isinstance(query, list):
            query = " ".join(str(x) for x in query)
        if not isinstance(query, str):
            query = str(query)

        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": max_results,
                "format": "json", "utf8": 1,
            }, headers=headers, timeout=10)
        r.raise_for_status()
        for sr in r.json().get("query", {}).get("search", []):
            title = sr.get("title", "")
            if not title:
                continue
            try:
                slug = quote_plus(title.replace(' ', '_'))
                sresp = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
                    headers=headers, timeout=8)
                if sresp.status_code != 200:
                    continue
                sdata = sresp.json()
                if sdata.get("type") == "disambiguation":
                    continue
                body = sdata.get("extract", "")
                if not body or len(body) < 30:
                    continue
                href = (sdata.get("content_urls", {})
                        .get("desktop", {})
                        .get("page", f"https://en.wikipedia.org/wiki/{slug}"))
                results.append({
                    "title": title, "body": body, "href": href,
                    "source": "wikipedia", "credibility": 85,
                })
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
    return results


def check_deps():
    d = {}
    for mod in ("yt_dlp", "whisper", "torch", "pydub",
                "librosa", "sklearn", "numpy"):
        try:
            __import__(mod)
            d[mod] = True
        except ImportError:
            d[mod] = False
    try:
        import faster_whisper  # noqa
        d["faster_whisper"] = True
    except ImportError:
        d["faster_whisper"] = False
    try:
        from duckduckgo_search import DDGS  # noqa
        d["duckduckgo"] = True
    except ImportError:
        d["duckduckgo"] = False
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True,
                       text=True, check=True)
        d["ffmpeg"] = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        d["ffmpeg"] = False
    # Diarization dependencies
    try:
        import scipy  # noqa
        d["scipy"] = True
    except ImportError:
        d["scipy"] = False
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # noqa
        d["speechbrain"] = True
    except ImportError:
        d["speechbrain"] = False
    try:
        from resemblyzer import VoiceEncoder  # noqa
        d["resemblyzer"] = True
    except ImportError:
        d["resemblyzer"] = False
    try:
        from pyannote.audio import Pipeline  # noqa
        d["pyannote"] = True
    except ImportError:
        d["pyannote"] = False
    return d


# ───────────────── Local Diarization ─────────────────

def _bridge_gaps(mask, max_gap_frames):
    """Fill short gaps in a boolean mask (e.g. bridging brief pauses in speech)."""
    result = mask.copy()
    i = 0
    while i < len(result):
        if not result[i]:
            j = i
            while j < len(result) and not result[j]:
                j += 1
            if 0 < (j - i) < max_gap_frames:
                result[i:j] = True
            i = j
        else:
            i += 1
    return result


def _smooth_boolean_mask(mask, window=15):
    """Smooth a boolean mask using moving majority vote (no scipy needed)."""
    result = np.zeros(len(mask), dtype=bool)
    half = window // 2
    for i in range(len(mask)):
        s = max(0, i - half)
        e = min(len(mask), i + half + 1)
        result[i] = np.mean(mask[s:e]) > 0.5
    return result


def _try_neural_diarize(audio_path, max_speakers=10, progress_cb=None):
    """
    Attempt neural diarization using pyannote.audio if available.
    Returns list of {start, end, speaker} dicts, or None if unavailable.
    Requires: pip install pyannote.audio  AND  a HuggingFace token with
    accepted model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        # Try to get HF token from env or cached credentials
        hf_token = os.environ.get("HF_TOKEN",
                    os.environ.get("HUGGINGFACE_TOKEN", ""))
        if not hf_token:
            try:
                from huggingface_hub import HfFolder
                hf_token = HfFolder.get_token() or ""
            except Exception:
                pass

        if not hf_token:
            logger.info("No HuggingFace token found — skipping pyannote")
            return None

        if progress_cb:
            progress_cb("Loading pyannote diarization pipeline…")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))

        if progress_cb:
            progress_cb("Running pyannote diarization…")

        diarization = pipeline(
            audio_path,
            min_num_speakers=2,
            max_num_speakers=max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        if segments and progress_cb:
            n_spk = len({s["speaker"] for s in segments})
            progress_cb(f"pyannote: {n_spk} speaker(s) detected ✓")

        return segments if segments else None

    except Exception as e:
        logger.info(f"Neural diarization unavailable: {e}")
        return None


def local_diarize(audio_path, max_speakers=10, seg_duration=1.0,
                  progress_cb=None, chunk_duration=300):
    """
    Improved local diarization with automatic speaker detection.

    Improvements over original:
    - Richer acoustic features (MFCC-20 + deltas + spectral shape)
    - Adaptive VAD with gap bridging and smoothing
    - Optional neural speaker embeddings (speechbrain / resemblyzer)
    - Global clustering for cross-chunk speaker consistency
    - Multi-metric optimal-K selection (silhouette + BIC + eigenvalue gap)
    - Single-speaker detection (avoids over-segmentation)
    - Post-processing: merge, smooth, absorb short segments
    """
    import librosa
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    if progress_cb:
        progress_cb("Loading audio metadata…")

    duration = librosa.get_duration(path=audio_path)
    sr = 16000
    hop = 512
    n_fft = 2048
    fps = max(1, int(seg_duration * sr / hop))
    total_chunks = max(1, int(np.ceil(duration / chunk_duration)))

    # ── Try to load neural speaker encoder ──
    speaker_encoder = None
    encoder_type = None
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        import torch as _torch
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_model_speechbrain",
            run_opts={"device": "cpu"},
        )
        encoder_type = "speechbrain"
        if progress_cb:
            progress_cb("Neural speaker encoder (SpeechBrain) loaded ✓")
    except Exception:
        try:
            from resemblyzer import VoiceEncoder
            speaker_encoder = VoiceEncoder()
            encoder_type = "resemblyzer"
            if progress_cb:
                progress_cb("Neural speaker encoder (Resemblyzer) loaded ✓")
        except Exception:
            if progress_cb:
                progress_cb("Acoustic features only (no neural encoder)")

    # ── Collect segment features across all chunks ──
    all_seg_info = []   # list of {"start": ..., "end": ...}
    all_acoustic = []   # list of acoustic feature vectors
    all_neural = []     # list of neural embedding vectors (or None)
    has_neural = False

    for ci in range(total_chunks):
        offset = ci * chunk_duration
        if progress_cb:
            progress_cb(f"Extracting features chunk {ci+1}/{total_chunks}…")

        y, _ = librosa.load(audio_path, sr=sr, mono=True,
                            offset=offset, duration=chunk_duration)
        if len(y) < sr:
            continue

        # ── Adaptive VAD ──
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]

        # Adaptive threshold using energy distribution
        sorted_rms = np.sort(rms)
        q25 = sorted_rms[max(0, len(sorted_rms) // 4)]
        q75 = sorted_rms[min(len(sorted_rms) - 1, 3 * len(sorted_rms) // 4)]
        rms_thresh = q25 + 0.15 * (q75 - q25)
        rms_thresh = max(rms_thresh, np.percentile(rms, 15))

        vad_mask = rms > rms_thresh

        # Bridge short gaps (< 300ms of silence)
        gap_frames = int(0.3 * sr / hop)
        vad_mask = _bridge_gaps(vad_mask, gap_frames)

        # Smooth with moving majority vote
        vad_mask = _smooth_boolean_mask(vad_mask, window=15)

        # ── Acoustic feature extraction ──
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20,
                                    n_fft=n_fft, hop_length=hop)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)

        feats = np.vstack([mfcc, d1, d2])  # 60 × T

        # Add spectral shape features
        try:
            sc = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop)
            sb = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop)
            sro = librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop)
            feats = np.vstack([feats, sc, sb, sro])  # 63 × T
        except Exception:
            pass

        T = feats.shape[1]
        n_segs = T // fps

        # ── Build per-segment features ──
        for i in range(n_segs):
            s, e = i * fps, (i + 1) * fps
            # VAD: require at least 40% speech frames in segment
            if s < len(vad_mask) and e <= len(vad_mask):
                if np.mean(vad_mask[s:e]) < 0.4:
                    continue
            elif s >= len(vad_mask):
                break

            chunk_feat = feats[:, s:e]

            # Statistical summaries: mean, std, median
            vec = np.concatenate([
                np.mean(chunk_feat, axis=1),
                np.std(chunk_feat, axis=1),
                np.median(chunk_feat, axis=1),
            ])

            start_time = offset + s * hop / sr
            end_time = offset + e * hop / sr

            all_seg_info.append({"start": start_time, "end": end_time})
            all_acoustic.append(vec)

            # Neural embedding for this segment
            if speaker_encoder is not None:
                s_sample = int(s * hop)
                e_sample = min(int(e * hop), len(y))
                seg_audio = y[s_sample:e_sample]
                # Pad short segments to at least 1 second
                if len(seg_audio) < sr:
                    seg_audio = np.pad(
                        seg_audio, (0, max(0, sr - len(seg_audio))))
                try:
                    if encoder_type == "speechbrain":
                        import torch
                        t_audio = torch.FloatTensor(seg_audio).unsqueeze(0)
                        emb = speaker_encoder.encode_batch(t_audio)
                        emb = emb.squeeze().detach().numpy()
                    else:  # resemblyzer
                        emb = speaker_encoder.embed_utterance(seg_audio)
                    all_neural.append(emb)
                    has_neural = True
                except Exception:
                    all_neural.append(None)
            else:
                all_neural.append(None)

        # Free chunk memory
        del y, mfcc, d1, d2, feats, rms, vad_mask
        gc.collect()

    # ── Handle empty result ──
    if not all_seg_info:
        return [{"start": 0, "end": duration, "speaker": "Speaker 1"}]

    # ── Build feature matrix ──
    X_acoustic = np.array(all_acoustic)
    X_acoustic = np.nan_to_num(X_acoustic, nan=0.0, posinf=0.0, neginf=0.0)
    X_acoustic = StandardScaler().fit_transform(X_acoustic)

    if has_neural:
        # Determine embedding dimension from first valid embedding
        emb_dim = 0
        for emb in all_neural:
            if emb is not None:
                emb_dim = len(emb)
                break

        if emb_dim > 0:
            neural_vecs = []
            for emb in all_neural:
                if emb is not None:
                    neural_vecs.append(emb)
                else:
                    neural_vecs.append(np.zeros(emb_dim))
            X_neural = np.array(neural_vecs)
            X_neural = np.nan_to_num(X_neural, nan=0.0, posinf=0.0, neginf=0.0)
            X_neural = StandardScaler().fit_transform(X_neural)
            # Weighted concatenation: neural embeddings are more discriminative
            X = np.hstack([X_acoustic * 0.3, X_neural * 0.7])
        else:
            X = X_acoustic
    else:
        X = X_acoustic

    # ── Determine optimal number of speakers ──
    max_k = min(max_speakers, len(X) - 1, max(2, len(X) // 3))

    if len(X) < 4 or max_k < 2:
        labels = np.zeros(len(X), dtype=int)
    else:
        best_n, best_sc = 2, -2.0
        scores_by_k = {}

        for k in range(2, max_k + 1):
            try:
                cl = AgglomerativeClustering(n_clusters=k, linkage='ward')
                lab = cl.fit_predict(X)
            except Exception:
                continue

            n_unique = len(set(lab))
            if n_unique < 2:
                continue

            # Silhouette score (sampled for large sets)
            try:
                sil = silhouette_score(
                    X, lab,
                    sample_size=min(len(X), 5000))
            except Exception:
                sil = 0.0

            # BIC-like penalty for model complexity
            n_pts, n_dim = len(X), X.shape[1]
            wcss = 0.0
            for c in range(k):
                mask = lab == c
                cnt = np.sum(mask)
                if cnt > 1:
                    wcss += np.sum(np.var(X[mask], axis=0)) * cnt
            bic_penalty = k * n_dim * np.log(n_pts) / n_pts if n_pts > 0 else 0

            combined = sil - 0.005 * bic_penalty
            scores_by_k[k] = combined

            if combined > best_sc:
                best_sc, best_n = combined, k

        # ── Eigenvalue gap check on graph Laplacian ──
        try:
            from scipy.spatial.distance import pdist, squareform
            D = squareform(pdist(X, metric='cosine'))
            sigma = np.median(D[D > 0]) if np.any(D > 0) else 1.0
            A = np.exp(-D ** 2 / (2 * sigma ** 2))
            np.fill_diagonal(A, 0)
            deg = np.sum(A, axis=1)
            safe_deg = np.maximum(deg, 1e-10)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(safe_deg))
            L_norm = D_inv_sqrt @ (np.diag(deg) - A) @ D_inv_sqrt
            eigvals = np.sort(np.linalg.eigvalsh(L_norm))[:max_k + 2]
            gaps = np.diff(eigvals)
            if len(gaps) >= 2:
                eig_k = np.argmax(gaps[1:]) + 2
                if 2 <= eig_k <= max_k and eig_k in scores_by_k:
                    # Eigenvalue gap suggests a different K — trust it if gap
                    # is significantly larger than the gap at best_n
                    if eig_k != best_n:
                        gap_at_eig = gaps[eig_k - 1] if eig_k - 1 < len(gaps) else 0
                        gap_at_best = gaps[best_n - 1] if best_n - 1 < len(gaps) else 0
                        if gap_at_eig > gap_at_best * 1.3:
                            best_n = eig_k
        except Exception:
            pass

        # ── Single-speaker detection ──
        if best_sc < 0.15:
            best_n = 1
            labels = np.zeros(len(X), dtype=int)
        else:
            labels = AgglomerativeClustering(
                n_clusters=best_n, linkage='ward'
            ).fit_predict(X)

    # ── Build output segments ──
    all_segments = []
    if len(labels) > 0:
        cur_label = int(labels[0])
        seg_s = all_seg_info[0]["start"]

        for i in range(1, len(labels)):
            if int(labels[i]) != cur_label:
                all_segments.append({
                    "start": seg_s,
                    "end": all_seg_info[i]["start"],
                    "speaker": f"Speaker {cur_label + 1}",
                })
                cur_label = int(labels[i])
                seg_s = all_seg_info[i]["start"]

        all_segments.append({
            "start": seg_s,
            "end": all_seg_info[-1]["end"],
            "speaker": f"Speaker {cur_label + 1}",
        })

    if not all_segments:
        return [{"start": 0, "end": duration, "speaker": "Speaker 1"}]

    # ── Post-processing ──

    # 1) Merge adjacent same-speaker segments with gap < 1.5s
    merged = [dict(all_segments[0])]
    for seg in all_segments[1:]:
        if (seg["speaker"] == merged[-1]["speaker"]
                and seg["start"] - merged[-1]["end"] < 1.5):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    # 2) Absorb very short segments (< 0.5s) into nearest neighbor
    final = []
    for i, seg in enumerate(merged):
        if seg["end"] - seg["start"] < 0.5:
            if final:
                final[-1]["end"] = seg["end"]
            elif i + 1 < len(merged):
                merged[i + 1]["start"] = seg["start"]
            else:
                final.append(seg)
        else:
            final.append(seg)

    # 3) Renumber speakers sequentially
    remap, ctr = {}, 1
    for seg in final:
        if seg["speaker"] not in remap:
            remap[seg["speaker"]] = f"Speaker {ctr}"
            ctr += 1
        seg["speaker"] = remap[seg["speaker"]]

    if progress_cb:
        n_spk = len({s["speaker"] for s in final})
        progress_cb(f"Diarization complete: {n_spk} speaker(s) detected")

    return final


# ═══════════════════════ Worker Threads ═══════════════════════

class VideoDownloadWorker(QThread):
    progress = Signal(int, str)
    info_ready = Signal(dict)
    finished = Signal(bool, str, str)

    def __init__(self, url, out_dir):
        super().__init__()
        self.url = url
        self.out_dir = out_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            import yt_dlp
            os.makedirs(self.out_dir, exist_ok=True)
            self.progress.emit(10, "Fetching info…")
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(self.url, download=False)
                vi = {
                    "title": info.get("title", "Unknown"),
                    "description": info.get("description", ""),
                    "duration": info.get("duration", 0),
                    "channel": info.get("channel", ""),
                    "upload_date": info.get("upload_date", ""),
                    "view_count": info.get("view_count", 0),
                    "categories": info.get("categories", []),
                    "url": self.url,
                }
                self.info_ready.emit(vi)
            if self._cancel:
                self.finished.emit(False, "Cancelled", "")
                return
            self.progress.emit(30, "Downloading audio…")
            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }],
                "outtmpl": os.path.join(self.out_dir, "video.%(ext)s"),
                "quiet": True, "no_warnings": True,
                "progress_hooks": [self._hook],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])
            audio_path = ""
            for f in Path(self.out_dir).glob("video.*"):
                if f.suffix in (".wav", ".mp3", ".m4a", ".ogg", ".flac"):
                    audio_path = str(f)
                    break
            if not audio_path:
                self.finished.emit(False, "Audio not found", "")
                return
            self.progress.emit(100, "Done!")
            self.finished.emit(True, "Downloaded ✓", audio_path)
        except Exception as e:
            self.finished.emit(False, f"Failed: {e}", "")

    def _hook(self, d):
        if self._cancel:
            raise Exception("Cancelled")
        if d["status"] == "downloading":
            try:
                p = int(float(d.get("_percent_str", "0%")
                              .strip().replace("%", "")))
                self.progress.emit(30 + int(p * 0.6), f"Downloading: {p}%")
            except ValueError:
                pass


class SubtitleFetchWorker(QThread):
    progress = Signal(int, str)
    subs_ready = Signal(list)
    finished = Signal(bool, str, str)

    def __init__(self, url, out_dir, lang="en"):
        super().__init__()
        self.url = url
        self.out_dir = out_dir
        self.lang = lang

    def run(self):
        try:
            import yt_dlp
            os.makedirs(self.out_dir, exist_ok=True)
            self.progress.emit(20, "Checking subs…")
            ydl_opts = {
                "writesubtitles": True, "writeautomaticsub": True,
                "subtitleslangs": [self.lang],
                "subtitlesformat": "vtt/srt",
                "skip_download": True, "quiet": True,
                "no_warnings": True,
                "outtmpl": os.path.join(self.out_dir, "video"),
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                subs = info.get("requested_subtitles", {}) or {}
            if not subs:
                self.finished.emit(False, "No subs", "")
                return
            self.progress.emit(60, "Parsing…")
            sp = ""
            for ext in (".vtt", ".srt"):
                for f in Path(self.out_dir).glob(f"video*{ext}"):
                    sp = str(f)
                    break
                if sp:
                    break
            if not sp:
                self.finished.emit(False, "Sub file missing", "")
                return
            with open(sp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            entries = parse_vtt(content) if ".vtt" in sp else parse_srt(content)
            if not entries:
                self.finished.emit(False, "Parse failed", "")
                return
            self.progress.emit(100, f"Found {len(entries)}!")
            self.subs_ready.emit(entries)
            self.finished.emit(True, f"Subs loaded ({len(entries)})", sp)
        except Exception as e:
            self.finished.emit(False, f"Sub fetch failed: {e}", "")


class TranscriptionWorker(QThread):
    progress = Signal(int, str)
    segment_ready = Signal(dict)
    transcript_ready = Signal(list)
    finished = Signal(bool, str)

    def __init__(self, audio_path, model_name, language="",
                 diarize=False, max_speakers=6, use_fw=False):
        super().__init__()
        self.audio = audio_path
        self.model_name = model_name
        self.lang = language or None
        self.dia = diarize
        self.max_spk = max_speakers
        self.use_fw = use_fw
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        model = None
        try:
            transcript = self._fw() if self.use_fw else self._whisper()
            if self._cancel:
                self.finished.emit(False, "Cancelled")
                return
            if self.dia and transcript:
                self.progress.emit(65, "Diarizing…")
                try:
                    # Try neural pipeline first (pyannote)
                    spk_segs = _try_neural_diarize(
                        self.audio, self.max_spk,
                        progress_cb=lambda msg: self.progress.emit(67, msg))
                    if spk_segs is None:
                        # Fall back to improved local diarization
                        spk_segs = local_diarize(
                            self.audio, self.max_spk,
                            progress_cb=lambda msg: self.progress.emit(67, msg))
                    transcript = self._assign(transcript, spk_segs)
                except Exception as e:
                    self.progress.emit(75, f"Dia failed: {e}")
            self.progress.emit(100, "Done!")
            self.transcript_ready.emit(transcript)
            self.finished.emit(True, "Complete")
        except Exception as e:
            self.finished.emit(False, f"Failed: {e}")
        finally:
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _whisper(self):
        import whisper
        self.progress.emit(5, f"Loading {self.model_name}…")
        model = whisper.load_model(self.model_name)
        self.progress.emit(15, "Transcribing…")
        opts = {"verbose": False, "word_timestamps": True}
        if self.lang:
            opts["language"] = self.lang
        result = model.transcribe(self.audio, **opts)
        self.progress.emit(55, "Building…")
        return [{
            "start": s["start"], "end": s["end"],
            "text": s["text"].strip(),
            "words": s.get("words", []),
            "speaker": "Speaker 1",
        } for s in result.get("segments", [])]

    def _fw(self):
        from faster_whisper import WhisperModel
        
        self.progress.emit(5, f"Loading FW {self.model_name}…")

        # Detect optimal settings for the specific GPU
        opt_dev, opt_ct = get_optimal_whisper_device_compute()
        
        model = None
        # Fallback chain: try detected optimal first, then fallback safely
        fallback_pairs = [
            (opt_dev, opt_ct),
            ("cuda", "int8_float16"),
            ("cuda", "int8"),
            ("cpu", "int8"),
        ]
        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for pair in fallback_pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)

        for dev, ct in unique_pairs:
            try:
                model = WhisperModel(self.model_name, device=dev, compute_type=ct)
                logger.info(f"Successfully loaded faster-whisper on {dev} with {ct}")
                break
            except Exception as e:
                logger.warning(f"FW load failed ({dev}/{ct}): {e}")
                continue

        if model is None:
            raise RuntimeError(
                "Could not load faster-whisper model on any device/compute type. "
                "Ensure ctranslate2 is installed correctly."
            )

        self.progress.emit(15, "Transcribing FW…")
        segs, _ = model.transcribe(self.audio, word_timestamps=True,
                                   language=self.lang or None)
        self.progress.emit(55, "Building…")
        
        tr = [{
            "start": s.start, "end": s.end,
            "text": s.text.strip(),
            "words": [{"start": w.start, "end": w.end, "word": w.word}
                      for w in (s.words or [])],
            "speaker": "Speaker 1",
        } for s in segs]

        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return tr

    @staticmethod
    def _assign(transcript, spk_segs):
        """Assign speakers to transcript segments based on temporal overlap."""
        if not spk_segs:
            for e in transcript:
                e["speaker"] = e.get("speaker", "Speaker 1")
            return transcript

        for e in transcript:
            best, best_ov = "Speaker 1", -1
            e_dur = e["end"] - e["start"]
            for ss in spk_segs:
                ov = max(0, min(e["end"], ss["end"]) - max(e["start"], ss["start"]))
                # Use proportional overlap so long segments don't dominate
                if e_dur > 0:
                    ov_ratio = ov / e_dur
                    if ov_ratio > best_ov:
                        best_ov, best = ov_ratio, ss["speaker"]
                elif ov > best_ov:
                    best_ov, best = ov, ss["speaker"]
            e["speaker"] = best
        return transcript


class SpeakerIDWorker(QThread):
    progress = Signal(int, str)
    ids_ready = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, transcript, video_info, ollama_url, llm_model):
        super().__init__()
        self.trans = transcript
        self.vi = video_info
        self.url = ollama_url
        self.model = llm_model

    def run(self):
        try:
            spk_txt = {}
            for e in self.trans:
                spk_txt.setdefault(e.get("speaker", "Speaker 1"), []).append(e["text"])
            for s in spk_txt:
                spk_txt[s] = " ".join(spk_txt[s])[:2000]
            self.progress.emit(20, "Analyzing…")
            parts = [f"Title: {self.vi.get('title', '')}\n"]
            for s, t in spk_txt.items():
                parts.append(f"[{s}]: {t}\n")
            parts.append('JSON: {"Speaker N": {"identity":"…","topics":["…"],"perspective":"…"}}')
            self.progress.emit(40, "Querying…")
            resp = ollama_request(self.url, self.model,
                                  "".join(parts), "JSON only.")
            if not resp:
                self.finished.emit(False, "Empty")
                return
            m = re.search(r'\{[\s\S]*\}', resp)
            ids = json.loads(m.group()) if m else {}
            self.ids_ready.emit(ids)
            self.progress.emit(100, "Done!")
            self.finished.emit(True, "Complete")
        except Exception as e:
            self.finished.emit(False, f"Failed: {e}")


class ClaimExtractionWorker(QThread):
    progress = Signal(int, str)
    claim_found = Signal(dict)
    finished = Signal(bool, str, list)

    def __init__(self, transcript, ollama_url, llm_model):
        super().__init__()
        self.trans = transcript
        self.url = ollama_url
        self.model = llm_model

    def run(self):
        try:
            self.progress.emit(10, "Prep…")
            full = "\n".join(
                f"[{e.get('speaker', '?')} @ {fmt_ts(e['start'])}]: {e['text']}"
                for e in self.trans)
            chunks = self._smart_chunk(full, 3000)
            all_claims = []
            for ci, ch in enumerate(chunks):
                self.progress.emit(
                    10 + int((ci / max(len(chunks), 1)) * 70),
                    f"Extracting {ci + 1}/{len(chunks)}…")
                sys_ = (
                    "Extract verifiable factual claims. Generate optimized "
                    "search query for each. JSON array: "
                    '[{"claim":"…","speaker":"…","timestamp":"…",'
                    '"topic":"…","verifiability":"high|medium|low",'
                    '"search_query":"…"}] or [].')
                resp = ollama_request(self.url, self.model,
                                      f"Transcript:\n\n{ch}", sys_)
                if resp:
                    try:
                        m = re.search(r'\[[\s\S]*\]', resp)
                        cl = json.loads(m.group()) if m else []
                        if isinstance(cl, list):
                            for c in cl:
                                if isinstance(c, dict) and "claim" in c:
                                    # ─── BUG FIX: Sanitize fields ─────────────────
                                    # Ensure all value fields are strings, not lists
                                    for key in ["claim", "search_query", "topic", "speaker", "timestamp"]:
                                        val = c.get(key)
                                        if isinstance(val, list):
                                            c[key] = " ".join(str(x) for x in val)
                                        elif val is None:
                                            c[key] = ""
                                        else:
                                            c[key] = str(val)
                                    # ─────────────────────────────────────────────
                                    
                                    c["id"] = len(all_claims) + 1
                                    if "search_query" not in c or not c["search_query"]:
                                        c["search_query"] = c["claim"]
                                    all_claims.append(c)
                                    self.claim_found.emit(c)
                    except json.JSONDecodeError:
                        pass
            self.progress.emit(100, "Done!")
            self.finished.emit(True, f"Extracted {len(all_claims)}", all_claims)
        except Exception as e:
            self.finished.emit(False, f"Failed: {e}", [])

    @staticmethod
    def _smart_chunk(text, max_len):
        lines = text.split('\n')
        chunks, current = [], ""
        for line in lines:
            if len(current) + len(line) + 1 > max_len and current:
                chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            chunks.append(current)
        return chunks


class ClaimSplitWorker(QThread):
    progress = Signal(int, str)
    split_ready = Signal(list)
    finished = Signal(bool, str, list)

    def __init__(self, claims, ollama_url, llm_model):
        super().__init__()
        self.claims = claims
        self.url = ollama_url
        self.model = llm_model

    def run(self):
        try:
            new_claims = []
            total = len(self.claims)
            for i, claim in enumerate(self.claims):
                self.progress.emit(
                    int((i / max(total, 1)) * 90),
                    f"Splitting {i + 1}/{total}…")
                resp = ollama_request(
                    self.url, self.model,
                    f'Original claim: "{claim.get("claim", "")}"\n'
                    f'Speaker: {claim.get("speaker", "")}',
                    "Split compound claims into individual verifiable parts. "
                    "Generate search query for each. JSON array: "
                    '[{"claim":"…","search_query":"…"}]. '
                    "Return unchanged if single claim.")
                if resp:
                    try:
                        m = re.search(r'\[[\s\S]*\]', resp)
                        parts = json.loads(m.group()) if m else []
                        if isinstance(parts, list) and len(parts) > 1:
                            for pi, part in enumerate(parts):
                                if isinstance(part, dict) and "claim" in part:
                                    nc = dict(claim)
                                    nc["claim"] = part["claim"].strip()
                                    nc["search_query"] = part.get(
                                        "search_query", part["claim"].strip())
                                    nc["split_from"] = claim.get("id", "?")
                                    nc["split_part"] = pi + 1
                                    new_claims.append(nc)
                    except Exception:
                        pass
            self.progress.emit(100, "Done!")
            self.split_ready.emit(new_claims)
            self.finished.emit(True, f"Split produced {len(new_claims)}", new_claims)
        except Exception as e:
            self.finished.emit(False, f"Failed: {e}", [])


# ───────────────── Restored FactCheckWorker ─────────────────

class FactCheckWorker(QThread):
    progress = Signal(int, str)
    result_ready = Signal(dict)
    finished = Signal(bool, str, list)

    def __init__(self, claims, ollama_url, llm_model,
                 max_concurrent=3, use_wikipedia=True):
        super().__init__()
        self.claims = claims
        self.url = ollama_url
        self.model = llm_model
        self.max_conc = max_concurrent
        self.use_wiki = use_wikipedia
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            results = []
            total = len(self.claims)
            with ThreadPoolExecutor(max_workers=self.max_conc) as executor:
                futures = {executor.submit(self._check_one, c): (i, c)
                           for i, c in enumerate(self.claims)}
                done = 0
                for fut in as_completed(futures):
                    if self._cancel:
                        break
                    try:
                        r = fut.result()
                    except Exception as e:
                        _, claim = futures[fut]
                        r = {**claim, "verdict": "Unverified",
                             "confidence": 0, "reasoning": str(e),
                             "correction": "", "key_evidence": [],
                             "sources": []}
                    
                    results.append(r)
                    self.result_ready.emit(r)
                    done += 1
                    self.progress.emit(int((done / total) * 100), f"Checked {done}/{total}")
            
            if self._cancel:
                self.finished.emit(False, "Cancelled", results)
            else:
                self.finished.emit(True, "Fact-check complete", results)
        except Exception as e:
            self.finished.emit(False, f"Failed: {e}", [])

    def _check_one(self, claim):
        query = claim.get("search_query") or claim.get("claim", "")
        
        # Defensive: Ensure query is string
        if isinstance(query, list):
            query = " ".join(str(x) for x in query)
        query = str(query)

        sources = []
        
        # 1. Wikipedia
        if self.use_wiki:
            try:
                wiki_res = wikipedia_search(query)
                if wiki_res:
                    sources.extend(wiki_res)
            except Exception as e:
                logger.warning(f"Wiki search error: {e}")

        # 2. Web Search (DDGS)
        try:
            web_res = web_search(query)
            if web_res:
                sources.extend(web_res)
        except Exception as e:
            logger.warning(f"Web search error: {e}")

        # Score credibility
        scored_sources = []
        for s in sources:
            if not s.get("href"):
                continue
            cred = score_source_credibility(
                s.get("href", ""), s.get("title", ""), s.get("body", ""),
                self.url, self.model
            )
            s["credibility"] = cred.get("score", 50)
            s["tier"] = cred.get("tier", "unknown")
            scored_sources.append(s)

        # Sort by credibility
        scored_sources.sort(key=lambda x: x.get("credibility", 0), reverse=True)
        
        # Limit context for LLM
        evidence_text = "\n".join(
            f"Source [{i+1}] ({s.get('credibility', 0)}%): {s.get('title', '')}\n{s.get('body', '')[:500]}"
            for i, s in enumerate(scored_sources[:5])
        )

        prompt = (
            f"Claim: \"{claim.get('claim', '')}\"\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Based on the evidence, determine the verdict (True, False, Partially True, Unverified, Misleading). "
            "Provide a confidence score (0-100), a brief reasoning, and a correction if False/Misleading. "
            "Respond strictly in JSON format: "
            '{"verdict": "...", "confidence": 0, "reasoning": "...", "correction": "..."}'
        )

        resp_text = ollama_request(self.url, self.model, prompt, system="You are a fact-checker AI. Respond only in JSON.")
        
        result = {**claim, "sources": scored_sources, "key_evidence": scored_sources[:3]}
        
        try:
            # Try to parse JSON from LLM response
            json_match = re.search(r'\{[\s\S]*\}', resp_text)
            if json_match:
                data = json.loads(json_match.group())
                result["verdict"] = data.get("verdict", "Unverified")
                result["confidence"] = data.get("confidence", 0)
                result["reasoning"] = data.get("reasoning", "")
                result["correction"] = data.get("correction", "")
            else:
                result["verdict"] = "Unverified"
                result["reasoning"] = "Failed to parse LLM response."
        except Exception:
            result["verdict"] = "Unverified"
            result["reasoning"] = "Error parsing fact-check result."

        return result


class ContradictionWorker(QThread):
    progress = Signal(int, str)
    contradiction_ready = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, results, ollama_url, llm_model):
        super().__init__()
        self.results = results
        self.url = ollama_url
        self.model = llm_model

    def run(self):
        try:
            self.progress.emit(20, "Grouping by speaker…")
            by_speaker = {}
            for r in self.results:
                spk = r.get("speaker", "Unknown")
                by_speaker.setdefault(spk, []).append(r)

            if len(by_speaker) < 2:
                self.contradiction_ready.emit(
                    "<p>Only one speaker found. Cannot check contradictions.</p>")
                self.finished.emit(True, "Done")
                return

            self.progress.emit(50, "Analyzing contradictions…")
            parts = [
                "Analyze these claims by different speakers. "
                "Identify any DIRECT contradictions between speakers. "
                "For each contradiction, state which speakers contradict, "
                "what they said, and which is likely correct.\n\n",
            ]
            for spk, claims in by_speaker.items():
                parts.append(f"== {spk} ==\n")
                for c in claims:
                    parts.append(
                        f"- Claim: {c.get('claim', '')} | "
                        f"Verdict: {c.get('verdict', '?')} | "
                        f"Confidence: {c.get('confidence', 0)}%\n")
                parts.append("\n")
            parts.append(
                "Format as HTML table with columns: Speaker A | Claim A | "
                "Speaker B | Claim B | Analysis. "
                "If no contradictions, say 'No contradictions found.'")

            resp = ollama_request(self.url, self.model,
                                  "".join(parts),
                                  "You are a debate analyst. Identify "
                                  "factual contradictions. Be concise.")
            self.contradiction_ready.emit(resp if resp else "<p>No analysis.</p>")
            self.progress.emit(100, "Done!")
            self.finished.emit(True, "Complete")
        except Exception as e:
            self.finished.emit(False, f"Failed: {e}")


class PullWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, url, name):
        super().__init__()
        self.url = url
        self.name = name

    def run(self):
        try:
            self.progress.emit(f"Pulling '{self.name}'…")
            ollama_pull_model(self.url, self.name,
                              lambda s: self.progress.emit(
                                  f"Pulling {self.name}: {s}"))
            self.finished.emit(True, f"Model '{self.name}' pulled ✓")
        except Exception as e:
            self.finished.emit(False, f"Pull failed: {e}")


class BatchPipelineWorker(QThread):
    progress = Signal(int, str)
    status = Signal(str, str)
    finished = Signal(bool, str, dict)

    def __init__(self, url, out_dir, ollama_url, whisper_model,
                 language, diarize, max_speakers, use_fw,
                 speaker_model, extract_model, check_model):
        super().__init__()
        self.url = url
        self.out_dir = out_dir
        self.ollama = ollama_url
        self.w_model = whisper_model
        self.lang = language
        self.dia = diarize
        self.max_spk = max_speakers
        self.use_fw = use_fw
        self.spk_model = speaker_model
        self.ext_model = extract_model
        self.chk_model = check_model
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        data = {"url": self.url, "video_info": {}, "transcript": [],
                "speaker_ids": {}, "claims": [], "results": []}
        try:
            self.status.emit(self.url, "Downloading")
            self.progress.emit(5, "Downloading…")
            audio_path, vi = self._download()
            if self._cancel:
                self.finished.emit(False, self.url, data)
                return
            data["video_info"] = vi
            self.status.emit(self.url, "Subs")
            self.progress.emit(20, "Checking subs…")
            transcript = self._try_subs()
            if not transcript:
                self.status.emit(self.url, "Transcribing")
                self.progress.emit(30, "Transcribing…")
                transcript = self._transcribe(audio_path)
            if self._cancel:
                self.finished.emit(False, self.url, data)
                return
            data["transcript"] = transcript
            self.status.emit(self.url, "Identifying")
            self.progress.emit(55, "Identifying…")
            data["speaker_ids"] = self._sid(transcript, vi)
            self.status.emit(self.url, "Extracting")
            self.progress.emit(70, "Extracting…")
            data["claims"] = self._extract(transcript)
            if data["claims"]:
                self.status.emit(self.url, "Fact-Checking")
                self.progress.emit(85, "Fact-checking…")
                data["results"] = self._fc(data["claims"])
            self._save(data)
            self.status.emit(self.url, "Complete")
            self.progress.emit(100, "Done!")
            self.finished.emit(True, self.url, data)
        except Exception as e:
            self.status.emit(self.url, "Error")
            self.finished.emit(False, self.url, data)

    def _download(self):
        import yt_dlp
        os.makedirs(self.out_dir, exist_ok=True)
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(self.url, download=False)
            vi = {
                "title": info.get("title", "Unknown"),
                "description": info.get("description", ""),
                "duration": info.get("duration", 0),
                "channel": info.get("channel", ""),
                "url": self.url,
            }
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "outtmpl": os.path.join(self.out_dir, "video.%(ext)s"),
            "quiet": True, "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])
        audio_path = ""
        for f in Path(self.out_dir).glob("video.*"):
            if f.suffix in (".wav", ".mp3", ".m4a", ".ogg", ".flac"):
                audio_path = str(f)
                break
        if not audio_path:
            raise Exception("Audio not found")
        return audio_path, vi

    def _try_subs(self):
        try:
            import yt_dlp
            ydl_opts = {
                "writesubtitles": True, "writeautomaticsub": True,
                "subtitleslangs": [self.lang or "en"],
                "subtitlesformat": "vtt/srt",
                "skip_download": True, "quiet": True,
                "outtmpl": os.path.join(self.out_dir, "video"),
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                subs = info.get("requested_subtitles", {}) or {}
            if not subs:
                return []
            sp = ""
            for ext in (".vtt", ".srt"):
                for f in Path(self.out_dir).glob(f"video*{ext}"):
                    sp = str(f)
                    break
                if sp:
                    break
            if not sp:
                return []
            with open(sp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return parse_vtt(content) if ".vtt" in sp else parse_srt(content)
        except Exception:
            return []

    def _transcribe(self, audio_path):
        if self.use_fw:
            from faster_whisper import WhisperModel
            import gc

            self.progress.emit(5, f"Loading FW {self.w_model}…")

            # Detect optimal settings for the specific GPU
            opt_dev, opt_ct = get_optimal_whisper_device_compute()
            
            model = None
            # Fallback chain: try detected optimal first, then fallback safely
            fallback_pairs = [
                (opt_dev, opt_ct),
                ("cuda", "int8_float16"),
                ("cuda", "int8"),
                ("cpu", "int8"),
            ]
            # Remove duplicates while preserving order
            seen = set()
            unique_pairs = []
            for pair in fallback_pairs:
                if pair not in seen:
                    seen.add(pair)
                    unique_pairs.append(pair)

            for dev, ct in unique_pairs:
                try:
                    model = WhisperModel(self.w_model, device=dev, compute_type=ct)
                    logger.info(f"Successfully loaded faster-whisper on {dev} with {ct}")
                    break
                except Exception as e:
                    logger.warning(f"FW load failed ({dev}/{ct}): {e}")
                    continue

            if model is None:
                raise RuntimeError(
                    "Could not load faster-whisper model on any device/compute type. "
                    "Ensure ctranslate2 is installed correctly."
                )

            segs, _ = model.transcribe(audio_path, word_timestamps=True,
                                       language=self.lang or None)
            tr = [{"start": s.start, "end": s.end, "text": s.text.strip(),
                   "words": [{"start": w.start, "end": w.end, "word": w.word}
                             for w in (s.words or [])],
                   "speaker": "Speaker 1"} for s in segs]

            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        else:
            import whisper
            import gc

            model = whisper.load_model(self.w_model)
            opts = {"verbose": False, "word_timestamps": True}
            if self.lang:
                opts["language"] = self.lang
            result = model.transcribe(audio_path, **opts)
            if result is None or "segments" not in result:
                logger.error("Transcription failed — is ffmpeg installed?")
                return []
            tr = [{"start": s["start"], "end": s["end"],
                   "text": s["text"].strip(), "words": s.get("words", []),
                   "speaker": "Speaker 1"}
                  for s in result.get("segments", [])]
            del model
            gc.collect()

        if self.dia and tr:
            spk = local_diarize(audio_path, self.max_spk)
            tr = TranscriptionWorker._assign(tr, spk)
        return tr

    def _sid(self, transcript, vi):
        if not transcript:
            return {}
        spk_txt = {}
        for e in transcript:
            spk_txt.setdefault(e.get("speaker", "Speaker 1"), []).append(e["text"])
        for s in spk_txt:
            spk_txt[s] = " ".join(spk_txt[s])[:2000]
        parts = [f"Title: {vi.get('title', '')}\n"]
        for s, t in spk_txt.items():
            parts.append(f"[{s}]: {t}\n")
        parts.append('JSON: {"Speaker N": {"identity":"…","topics":["…"],"perspective":"…"}}')
        resp = ollama_request(self.ollama, self.spk_model,
                              "".join(parts), "JSON only.")
        if resp:
            try:
                m = re.search(r'\{[\s\S]*\}', resp)
                return json.loads(m.group()) if m else {}
            except Exception:
                pass
        return {}

    def _extract(self, transcript):
        if not transcript:
            return []
        full = "\n".join(
            f"[{e.get('speaker', '?')}]: {e['text']}" for e in transcript)
        chunks = ClaimExtractionWorker._smart_chunk(full, 3000)
        all_c = []
        for ch in chunks:
            sys_ = ('Extract claims + search queries. JSON array: '
                    '[{"claim":"…","speaker":"…","search_query":"…"}] or [].')
            resp = ollama_request(self.ollama, self.ext_model,
                                  f"Transcript:\n\n{ch}", sys_)
            if resp:
                try:
                    m = re.search(r'\[[\s\S]*\]', resp)
                    cl = json.loads(m.group()) if m else []
                    if isinstance(cl, list):
                        for c in cl:
                            if isinstance(c, dict) and "claim" in c:
                                c["id"] = len(all_c) + 1
                                if "search_query" not in c or not c["search_query"]:
                                    c["search_query"] = c["claim"]
                                all_c.append(c)
                except json.JSONDecodeError:
                    pass
        return all_c

    def _fc(self, claims):
        results = []
        for i, claim in enumerate(claims):
            if self._cancel:
                break
            self.progress.emit(
                85 + int((i / max(len(claims), 1)) * 10),
                f"Checking {i + 1}/{len(claims)}…")
            ct = claim.get("claim", "")
            sq = claim.get("search_query", ct)

            wiki_results = []
            try:
                entities = wikipedia_extract_entities(ct, self.ollama, self.chk_model)
                for ent in entities[:2]:
                    wiki_results.extend(wikipedia_search(ent, 2))
                if not wiki_results:
                    wiki_results = wikipedia_search(sq, 3)
            except Exception:
                pass

            ddg_results = web_search(sq, 5)
            all_sources_raw = wiki_results + ddg_results

            scored_sources = []
            for src in all_sources_raw:
                cred = score_source_credibility(
                    src.get("href", ""), src.get("title", ""),
                    src.get("body", ""), self.ollama, self.chk_model)
                scored_sources.append({**src, **cred})

            weighted = compute_weighted_verdict(scored_sources)

            wiki_ctx = ""
            if wiki_results:
                wiki_ctx = "\n=== WIKIPEDIA ===\n"
                for r in wiki_results:
                    wiki_ctx += (f"Title: {r.get('title', 'N/A')}\n"
                                 f"Excerpt: {r.get('body', 'N/A')}\n\n")

            ddg_ctx = "\n=== WEB ===\n"
            for r in scored_sources:
                ddg_ctx += (f"Title: {r.get('title', 'N/A')}\n"
                            f"Credibility: {r.get('score', 50)}/100\n"
                            f"Excerpt: {r.get('body', 'N/A')}\n\n")

            sys_ = ("Fact-checker. JSON only: "
                    '{"verdict":"True|False|Partially True|Unverified|Misleading",'
                    '"confidence":0-100,"reasoning":"…","correction":"…",'
                    '"key_evidence":["…"],"sources":["…"]}')
            prompt = (f'Claim: "{ct}"\nSpeaker: {claim.get("speaker", "?")}\n\n'
                      f"{wiki_ctx}\n{ddg_ctx}\nVerify.")

            resp = ollama_request(self.ollama, self.chk_model, prompt, sys_)

            result = {**claim, "verdict": "Unverified", "confidence": 0,
                      "reasoning": "", "correction": "", "key_evidence": [],
                      "sources": []}
            if resp:
                try:
                    m = re.search(r'\{[\s\S]*\}', resp)
                    if m:
                        p = json.loads(m.group())
                        for k in ("verdict", "confidence", "reasoning",
                                  "correction", "key_evidence", "sources"):
                            if k in p:
                                result[k] = p[k]
                except Exception:
                    result["reasoning"] = resp

            if not result["sources"]:
                result["sources"] = [r["href"] for r in all_sources_raw
                                     if r.get("href")]
            results.append(result)
        return results

    def _save(self, data):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            title = safe_filename(data.get("video_info", {}).get("title", "result"))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save JSON
            json_path = os.path.join(self.out_dir, f"{title}_{ts}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            # Save CSV
            csv_path = os.path.join(self.out_dir, f"{title}_{ts}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "id", "claim", "speaker", "verdict", "confidence",
                    "reasoning", "correction", "key_evidence", "sources",
                    "search_query"])
                writer.writeheader()
                for r in data.get("results", []):
                    row = {k: str(v) if isinstance(v, (list, dict)) else v
                           for k, v in r.items()
                           if k in writer.fieldnames}
                    writer.writerow(row)

            logger.info(f"Saved: {json_path}, {csv_path}")
        except Exception as e:
            logger.error(f"Save failed: {e}")

# ═════════════ Dialogs ═════════════

class SubtitleChoiceDialog(QDialog):
    def __init__(self, num_entries, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YouTube Subtitles Found")
        self.setMinimumWidth(420)
        self.choice = "whisper"

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            f"<b>✅ Found {num_entries} subtitle entries!</b><br>"
            "How to proceed?"))
        self.rb_subs = QRadioButton("Use YouTube subtitles (fast)")
        self.rb_both = QRadioButton("Both: Subs + Whisper (verify)")
        self.rb_whisper = QRadioButton("Whisper only (most accurate)")
        self.rb_subs.setChecked(True)
        lay.addWidget(self.rb_subs)
        lay.addWidget(self.rb_both)
        lay.addWidget(self.rb_whisper)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _accept(self):
        if self.rb_subs.isChecked():
            self.choice = "subs"
        elif self.rb_both.isChecked():
            self.choice = "both"
        else:
            self.choice = "whisper"
        self.accept()


class CustomClaimDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Custom Claim")
        self.setMinimumWidth(500)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Enter the factual claim you heard:"))
        self.claim_edit = QLineEdit()
        self.claim_edit.setPlaceholderText(
            "e.g., The population of France is 70 million")
        lay.addWidget(self.claim_edit)
        lay.addWidget(QLabel("Speaker (optional):"))
        self.speaker_edit = QLineEdit()
        self.speaker_edit.setPlaceholderText("Speaker 1")
        lay.addWidget(self.speaker_edit)
        lay.addWidget(QLabel("Optimized Search Query (optional):"))
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("France population 2024")
        lay.addWidget(self.query_edit)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def get_claim(self):
        c = self.claim_edit.text().strip()
        if not c:
            return None
        return {
            "claim": c,
            "speaker": self.speaker_edit.text().strip() or "Custom",
            "timestamp": "Manual",
            "topic": "Custom",
            "verifiability": "high",
            "search_query": self.query_edit.text().strip() or c,
        }


# ═══════════════════════ Main Window ═══════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1400, 950)

        self.audio_path = None
        self.transcript = []
        self.claims = []
        self.results = []
        self.video_info = {}
        self.speaker_ids = {}
        self.results_map = {}
        self.worker = None
        self.yt_transcript = []
        self.batch_queue = []
        self.batch_results = {}
        self.batch_worker = None

        self._s = QSettings("FactChecker", APP_NAME)
        self.out_dir = self._s.value("out_dir", DEFAULT_OUT_DIR, str)
        self.ollama = self._s.value("ollama", DEFAULT_OLLAMA, str)
        self.spk_model = self._s.value("spk_model", FAST_LLM, str)
        self.ext_model = self._s.value("ext_model", FAST_LLM, str)
        self.chk_model = self._s.value("chk_model", RECOMMENDED_LLM, str)
        self.use_fw = self._s.value("use_fw", True, bool)

        self.deps = check_deps()
        self._build()
        self._dep_warn()

    # ────────────────── Build UI ──────────────────

    def _build(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)
        root.setContentsMargins(8, 8, 8, 8)

        sp = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        sp.addWidget(self.tabs)

        lg = QGroupBox("Log")
        ll = QVBoxLayout(lg)
        self.log_ = QTextEdit()
        self.log_.setReadOnly(True)
        self.log_.setMaximumHeight(160)
        self.log_.setFont(QFont("Consolas", 9))
        ll.addWidget(self.log_)
        sp.addWidget(lg)

        sp.setSizes([720, 160])
        root.addWidget(sp)

        self.statusBar().showMessage("Ready")
        self.pbar = QProgressBar()
        self.pbar.setMaximumWidth(320)
        self.pbar.setVisible(False)
        self.statusBar().addPermanentWidget(self.pbar)

        self._build_setup()
        self._build_video()
        self._build_transcribe()
        self._build_fc()
        self._build_batch()
        self._build_report()

    def _build_setup(self):
        t = QWidget()
        lay = QVBoxLayout(t)

        # Dependencies
        dg = QGroupBox("Dependency Status")
        gl = QGridLayout(dg)
        items = [
            ("yt-dlp", "yt_dlp", "YouTube"),
            ("Whisper", "whisper", "Standard"),
            ("Faster-Whisper", "faster_whisper", "Fast (4x)"),
            ("PyTorch", "torch", "ML"),
            ("librosa", "librosa", "Audio/Dia"),
            ("scikit-learn", "sklearn", "Clustering"),
            ("DuckDuckGo", "duckduckgo", "Search"),
            ("ffmpeg", "ffmpeg", "System"),
        ]
        self.dlbl = {}
        for i, (n, k, d) in enumerate(items):
            gl.addWidget(QLabel(f"<b>{n}</b>"), i, 0)
            gl.addWidget(QLabel(f"<i>{d}</i>"), i, 1)
            l = QLabel()
            self.dlbl[k] = l
            gl.addWidget(l, i, 2)
        self._refresh_deps()
        gl.setColumnStretch(1, 1)
        lay.addWidget(dg)

        # Ollama
        og = QGroupBox("Ollama")
        ol = QGridLayout(og)
        ol.addWidget(QLabel("URL:"), 0, 0)
        self.o_edit = QLineEdit(self.ollama)
        self.o_edit.editingFinished.connect(
            lambda: self._sv("ollama", self.o_edit.text()))
        ol.addWidget(self.o_edit, 0, 1)
        self.conn_btn = QPushButton("Check")
        self.conn_btn.clicked.connect(self._chk_ollama)
        ol.addWidget(self.conn_btn, 0, 2)
        ol.setColumnStretch(1, 1)
        lay.addWidget(og)

        # Per-task models
        mg = QGroupBox("Per-Task LLM Models")
        ml = QGridLayout(mg)
        ml.addWidget(QLabel("Speaker ID:"), 0, 0)
        self.spk_combo = QComboBox()
        self.spk_combo.setEditable(True)
        self.spk_combo.addItems(LLM_MODELS)
        self.spk_combo.setCurrentText(self.spk_model)
        self.spk_combo.currentTextChanged.connect(
            lambda t: self._sv("spk_model", t))
        ml.addWidget(self.spk_combo, 0, 1)

        ml.addWidget(QLabel("Extraction:"), 1, 0)
        self.ext_combo = QComboBox()
        self.ext_combo.setEditable(True)
        self.ext_combo.addItems(LLM_MODELS)
        self.ext_combo.setCurrentText(self.ext_model)
        self.ext_combo.currentTextChanged.connect(
            lambda t: self._sv("ext_model", t))
        ml.addWidget(self.ext_combo, 1, 1)

        ml.addWidget(QLabel("Fact-Check:"), 2, 0)
        self.chk_combo = QComboBox()
        self.chk_combo.setEditable(True)
        self.chk_combo.addItems(LLM_MODELS)
        self.chk_combo.setCurrentText(self.chk_model)
        self.chk_combo.currentTextChanged.connect(
            lambda t: self._sv("chk_model", t))
        ml.addWidget(self.chk_combo, 2, 1)

        self.pull_btn = QPushButton("Pull Model")
        self.pull_btn.clicked.connect(self._pull)
        ml.addWidget(self.pull_btn, 0, 2)
        self.ref_btn = QPushButton("Refresh")
        self.ref_btn.clicked.connect(self._chk_ollama)
        ml.addWidget(self.ref_btn, 1, 2)
        ml.setColumnStretch(1, 1)
        lay.addWidget(mg)

        # Engine
        fwg = QGroupBox("Engine")
        fwl = QHBoxLayout(fwg)
        self.fw_chk = QCheckBox("Use faster-whisper (4x faster, less RAM)")
        self.fw_chk.setChecked(self.use_fw)
        self.fw_chk.toggled.connect(lambda v: self._sv("use_fw", v))
        fwl.addWidget(self.fw_chk)
        lay.addWidget(fwg)

        # Output dir
        odg = QGroupBox("Output Directory")
        odl = QHBoxLayout(odg)
        self.out_edit = QLineEdit(self.out_dir)
        self.out_edit.editingFinished.connect(
            lambda: self._sv("out_dir", self.out_edit.text()))
        odl.addWidget(self.out_edit)
        br = QPushButton("Browse")
        br.clicked.connect(self._br_out)
        odl.addWidget(br)
        lay.addWidget(odg)

        lay.addStretch()
        self.tabs.addTab(t, "⚙️ Setup")

    def _build_video(self):
        t = QWidget()
        lay = QVBoxLayout(t)

        ug = QGroupBox("YouTube URL")
        ul = QHBoxLayout(ug)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://youtube.com/watch?v=…")
        self.url_edit.returnPressed.connect(self._dl)
        ul.addWidget(self.url_edit)
        db = QPushButton("🔍 Fetch & Download")
        db.setStyleSheet("font-weight:bold;padding:8px;")
        db.clicked.connect(self._dl)
        ul.addWidget(db)
        lay.addWidget(ug)

        ig = QGroupBox("Video Info")
        il = QGridLayout(ig)
        self.vlbl = {}
        for i, f in enumerate(("Title", "Channel", "Duration",
                                "Upload Date", "Views", "Categories")):
            il.addWidget(QLabel(f"<b>{f}:</b>"), i, 0)
            v = QLabel("-")
            v.setWordWrap(True)
            self.vlbl[f] = v
            il.addWidget(v, i, 1)
        il.setColumnStretch(1, 1)
        lay.addWidget(ig)

        dg = QGroupBox("Description")
        dl = QVBoxLayout(dg)
        self.desc = QTextEdit()
        self.desc.setReadOnly(True)
        self.desc.setMaximumHeight(120)
        dl.addWidget(self.desc)
        lay.addWidget(dg)

        ag = QGroupBox("Audio")
        al = QHBoxLayout(ag)
        self.a_lbl = QLabel("No audio yet")
        al.addWidget(self.a_lbl)
        lay.addWidget(ag)

        lg = QGroupBox("Local Audio")
        ll = QHBoxLayout(lg)
        self.loc_edit = QLineEdit()
        self.loc_edit.setPlaceholderText("Path…")
        ll.addWidget(self.loc_edit)
        lb = QPushButton("Browse")
        lb.clicked.connect(self._br_loc)
        ll.addWidget(lb)
        lo = QPushButton("Load")
        lo.clicked.connect(self._load_loc)
        ll.addWidget(lo)
        lay.addWidget(lg)

        lay.addStretch()
        self.tabs.addTab(t, "🎬 Video")

    def _build_transcribe(self):
        t = QWidget()
        lay = QVBoxLayout(t)

        sg = QGroupBox("Settings")
        sl = QGridLayout(sg)
        sl.addWidget(QLabel("Model:"), 0, 0)
        self.w_combo = QComboBox()
        self.w_combo.addItems(WHISPER_MODELS)
        self.w_combo.setCurrentText("base")
        sl.addWidget(self.w_combo, 0, 1)

        sl.addWidget(QLabel("Lang:"), 0, 2)
        self.lang_combo = QComboBox()
        self.lang_combo.setEditable(True)
        self.lang_combo.addItems(
            ["", "en", "es", "fr", "de", "it", "pt",
             "ru", "zh", "ja", "ko", "ar"])
        sl.addWidget(self.lang_combo, 0, 3)

        self.dia_chk = QCheckBox("Speaker Diarization")
        self.dia_chk.setChecked(True)
        sl.addWidget(self.dia_chk, 1, 0, 1, 2)

        sl.addWidget(QLabel("Max Spk:"), 1, 2)
        self.max_spk = QSpinBox()
        self.max_spk.setRange(2, 10)
        self.max_spk.setValue(4)
        sl.addWidget(self.max_spk, 1, 3)

        self.sub_chk = QCheckBox("Check YouTube subs first")
        self.sub_chk.setChecked(True)
        sl.addWidget(self.sub_chk, 2, 0, 1, 2)

        btns = QHBoxLayout()
        self.tr_btn = QPushButton("🎙️ Transcribe")
        self.tr_btn.setStyleSheet("font-weight:bold;padding:10px;")
        self.tr_btn.clicked.connect(self._transcribe)
        btns.addWidget(self.tr_btn)
        self.sid_btn = QPushButton("👤 Identify Speakers")
        self.sid_btn.setEnabled(False)
        self.sid_btn.clicked.connect(self._sid)
        btns.addWidget(self.sid_btn)
        self.cn_btn = QPushButton("❌ Cancel")
        self.cn_btn.setEnabled(False)
        self.cn_btn.clicked.connect(self._cancel)
        btns.addWidget(self.cn_btn)
        sl.addLayout(btns, 3, 0, 1, 4)
        lay.addWidget(sg)

        tg = QGroupBox("Transcript")
        tl = QVBoxLayout(tg)
        self.tr_text = QTextEdit()
        self.tr_text.setReadOnly(True)
        self.tr_text.setFont(QFont("Consolas", 10))
        tl.addWidget(self.tr_text)
        lay.addWidget(tg)

        spg = QGroupBox("Speakers")
        spl = QVBoxLayout(spg)
        self.sp_text = QTextEdit()
        self.sp_text.setReadOnly(True)
        self.sp_text.setMaximumHeight(200)
        spl.addWidget(self.sp_text)
        lay.addWidget(spg)

        self.tabs.addTab(t, "📝 Transcribe")

    def _build_fc(self):
        t = QWidget()
        lay = QVBoxLayout(t)

        cg = QGroupBox("Controls")
        cl = QHBoxLayout(cg)
        self.ext_btn = QPushButton("🔍 Extract")
        self.ext_btn.setEnabled(False)
        self.ext_btn.clicked.connect(self._ext)
        cl.addWidget(self.ext_btn)
        self.split_btn = QPushButton("✂️ Split Claims")
        self.split_btn.setEnabled(False)
        self.split_btn.clicked.connect(self._split_claims)
        cl.addWidget(self.split_btn)
        self.custom_btn = QPushButton("➕ Add Custom Claim")
        self.custom_btn.setEnabled(False)
        self.custom_btn.clicked.connect(self._add_custom_claim)
        cl.addWidget(self.custom_btn)
        self.fc_btn = QPushButton("✅ Fact-Check Checked")
        self.fc_btn.setEnabled(False)
        self.fc_btn.clicked.connect(self._fc)
        cl.addWidget(self.fc_btn)
        self.fc_cn = QPushButton("❌ Cancel")
        self.fc_cn.setEnabled(False)
        self.fc_cn.clicked.connect(self._cancel)
        cl.addWidget(self.fc_cn)
        cl.addWidget(QLabel("Conc:"))
        self.conc_spin = QSpinBox()
        self.conc_spin.setRange(1, 8)
        self.conc_spin.setValue(3)
        cl.addWidget(self.conc_spin)
        cl.addStretch()
        lay.addWidget(cg)

        sp = QSplitter(Qt.Orientation.Horizontal)
        cgrp = QGroupBox("Claims (Check to include in Fact-Check)")
        cll = QVBoxLayout(cgrp)
        self.ctbl = QTableWidget()
        self.ctbl.setColumnCount(6)
        self.ctbl.setHorizontalHeaderLabels(
            ["✅", "#", "Speaker", "Claim", "Topic", "Verdict"])
        self.ctbl.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch)
        self.ctbl.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.ctbl.setAlternatingRowColors(True)
        self.ctbl.cellClicked.connect(self._claim_click)
        cll.addWidget(self.ctbl)
        sp.addWidget(cgrp)

        dgrp = QGroupBox("Detail")
        dll = QVBoxLayout(dgrp)
        self.det = QTextEdit()
        self.det.setReadOnly(True)
        self.det.setFont(QFont("Segoe UI", 10))
        dll.addWidget(self.det)
        sp.addWidget(dgrp)

        sp.setSizes([600, 500])
        lay.addWidget(sp)
        self.tabs.addTab(t, "🔎 Fact Check")

    def _build_batch(self):
        t = QWidget()
        lay = QVBoxLayout(t)

        ig = QGroupBox("Batch URLs")
        il = QVBoxLayout(ig)
        self.batch_urls = QPlainTextEdit()
        self.batch_urls.setMaximumHeight(100)
        il.addWidget(self.batch_urls)
        btns = QHBoxLayout()
        add_btn = QPushButton("➕ Add")
        add_btn.clicked.connect(self._batch_add)
        btns.addWidget(add_btn)
        rm_btn = QPushButton("🗑️ Remove")
        rm_btn.clicked.connect(self._batch_remove)
        btns.addWidget(rm_btn)
        clear_btn = QPushButton("🧹 Clear")
        clear_btn.clicked.connect(self._batch_clear)
        btns.addWidget(clear_btn)
        btns.addStretch()
        il.addLayout(btns)
        lay.addWidget(ig)

        qg = QGroupBox("Queue")
        ql = QVBoxLayout(qg)
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(4)
        self.batch_table.setHorizontalHeaderLabels(
            ["URL / Title", "Status", "Claims", "Verdicts"])
        self.batch_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.cellDoubleClicked.connect(self._batch_dblclick)
        ql.addWidget(self.batch_table)
        bc = QHBoxLayout()
        self.batch_start = QPushButton("▶️ Start")
        self.batch_start.setStyleSheet("font-weight:bold;padding:10px;")
        self.batch_start.clicked.connect(self._batch_go)
        bc.addWidget(self.batch_start)
        self.batch_cancel = QPushButton("❌ Cancel")
        self.batch_cancel.setEnabled(False)
        self.batch_cancel.clicked.connect(self._batch_stop)
        bc.addWidget(self.batch_cancel)
        self.batch_prog = QProgressBar()
        bc.addWidget(self.batch_prog)
        ql.addLayout(bc)
        lay.addWidget(qg)

        self.tabs.addTab(t, "📦 Batch")

    def _build_report(self):
        t = QWidget()
        lay = QVBoxLayout(t)

        bl = QHBoxLayout()
        gen = QPushButton("📄 Generate HTML")
        gen.setStyleSheet("font-weight:bold;padding:10px;")
        gen.clicked.connect(self._gen)
        bl.addWidget(gen)
        sv = QPushButton("💾 Save HTML")
        sv.clicked.connect(self._save_html)
        bl.addWidget(sv)
        csv_btn = QPushButton("📊 Export CSV")
        csv_btn.clicked.connect(self._export_csv)
        bl.addWidget(csv_btn)
        pdf_btn = QPushButton("📑 Export PDF")
        pdf_btn.clicked.connect(self._export_pdf)
        bl.addWidget(pdf_btn)
        bl.addStretch()
        lay.addLayout(bl)

        bl2 = QHBoxLayout()
        sp_btn = QPushButton("💾 Save Project")
        sp_btn.clicked.connect(self._save_project)
        bl2.addWidget(sp_btn)
        lp_btn = QPushButton("📂 Load Project")
        lp_btn.clicked.connect(self._load_project)
        bl2.addWidget(lp_btn)
        cont_btn = QPushButton("⚡ Contradiction Matrix")
        cont_btn.clicked.connect(self._contradiction_matrix)
        bl2.addWidget(cont_btn)
        bl2.addStretch()
        lay.addLayout(bl2)

        self.rpt = QTextEdit()
        self.rpt.setReadOnly(True)
        lay.addWidget(self.rpt)
        self.tabs.addTab(t, "📊 Report")

    # ────────────────── Settings ──────────────────

    def _sv(self, k, v):
        setattr(self, k, v)
        self._s.setValue(k, v)

    def _br_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output")
        if d:
            self.out_edit.setText(d)
            self._sv("out_dir", d)

    def _refresh_deps(self):
        for k, l in self.dlbl.items():
            ok = self.deps.get(k, False)
            l.setText("✅" if ok else "❌")
            l.setStyleSheet(
                "color:green;font-weight:bold" if ok
                else "color:red;font-weight:bold")

    def _dep_warn(self):
        miss = [k for k, v in self.deps.items() if not v]
        if miss:
            self.log(f"⚠️ Missing: {', '.join(miss)}")

    def _chk_ollama(self):
        try:
            r = requests.get(f"{self.ollama}/api/tags", timeout=5)
            if r.status_code == 200:
                names = [m["name"] for m in r.json().get("models", [])]
                self.conn_btn.setText("✅ OK")
                self.conn_btn.setStyleSheet("color:green;")
                for cb in (self.spk_combo, self.ext_combo, self.chk_combo):
                    p = cb.currentText()
                    cb.clear()
                    cb.addItems(names)
                    if p in names:
                        cb.setCurrentText(p)
                self.log(f"✅ Ollama: {', '.join(names)}")
            else:
                self.conn_btn.setText("❌ Err")
                self.conn_btn.setStyleSheet("color:red;")
        except Exception:
            self.conn_btn.setText("❌ Off")
            self.conn_btn.setStyleSheet("color:red;")

    def _pull(self):
        name = self.spk_combo.currentText()
        if not name:
            return
        self.pull_btn.setEnabled(False)
        self.worker = PullWorker(self.ollama, name)
        self.worker.progress.connect(lambda s: self.log(s))
        self.worker.finished.connect(self._pull_done)
        self.worker.start()

    def _pull_done(self, ok, msg):
        self.pull_btn.setEnabled(True)
        self.log(f"{'✅' if ok else '❌'} {msg}")
        if ok:
            self._chk_ollama()

    # ────────────────── Video ──────────────────

    def _dl(self):
        url = self.url_edit.text().strip()
        if not url:
            return
        self.log(f"Downloading: {url}")
        self._busy(True)
        self.worker = VideoDownloadWorker(url, self.out_dir)
        self.worker.progress.connect(self._prog)
        self.worker.info_ready.connect(self._vi)
        self.worker.finished.connect(self._dl_done)
        self.worker.start()

    def _vi(self, info):
        self.video_info = info
        self.vlbl["Title"].setText(info.get("title", "-"))
        self.vlbl["Channel"].setText(info.get("channel", "-"))
        self.vlbl["Duration"].setText(fmt_ts(info.get("duration", 0)))
        self.vlbl["Upload Date"].setText(info.get("upload_date", "-"))
        self.vlbl["Views"].setText(f"{info.get('view_count', 0):,}")
        self.vlbl["Categories"].setText(
            ", ".join(info.get("categories", [])))
        self.desc.setText(info.get("description", ""))

    def _dl_done(self, ok, msg, path):
        self._busy(False)
        self.log(f"{'✅' if ok else '❌'} {msg}")
        if ok and path:
            self.audio_path = path
            self.a_lbl.setText(f"✅ {path}")

    def _br_loc(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Audio", "",
            "Audio (*.wav *.mp3 *.m4a *.flac *.ogg);;All (*)")
        if f:
            self.loc_edit.setText(f)

    def _load_loc(self):
        p = self.loc_edit.text().strip()
        if os.path.isfile(p):
            self.audio_path = p
            self.a_lbl.setText(f"✅ {p}")
            self.video_info = {"title": Path(p).stem, "channel": "Local"}
            self.vlbl["Title"].setText(Path(p).stem)
        else:
            QMessageBox.warning(self, "Warning", "File not found")

    # ────────────────── Transcription ──────────────────

    def _update_speaker_combo(self, transcript):
        """Populate speaker filter combo with detected speakers."""
        speakers = sorted(set(
            e.get("speaker", "Speaker 1") for e in transcript))
        self.speaker_combo.blockSignals(True)
        self.speaker_combo.clear()
        self.speaker_combo.addItem("All Speakers")
        for s in speakers:
            self.speaker_combo.addItem(s)
        self.speaker_combo.blockSignals(False)

    def _transcribe(self):
        if not self.audio_path:
            return
        url = self.url_edit.text().strip()
        if self.sub_chk.isChecked() and url and self.deps.get("yt_dlp"):
            self.log("Checking subs…")
            self.yt_transcript = []
            self._busy(True)
            self.worker = SubtitleFetchWorker(
                url, self.out_dir, self.lang_combo.currentText() or "en")
            self.worker.progress.connect(self._prog)
            self.worker.subs_ready.connect(
                lambda e: setattr(self, 'yt_transcript', e))
            self.worker.finished.connect(self._subs_done)
            self.worker.start()
            return
        self._do_whisper()

    def _subs_done(self, ok, msg, path):
        self._busy(False)
        if self.yt_transcript:
            dlg = SubtitleChoiceDialog(len(self.yt_transcript), self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                if dlg.choice == "subs":
                    self.transcript = self.yt_transcript
                    self._render_tr()
                    self.sid_btn.setEnabled(True)
                    self.ext_btn.setEnabled(True)
                    self.custom_btn.setEnabled(True)
                    self.log("✅ Subs loaded")
                    return
                elif dlg.choice == "both":
                    self._do_whisper(verify=self.yt_transcript)
                    return
        self._do_whisper()

    def _do_whisper(self, verify=None):
        model = self.w_combo.currentText()
        lang = self.lang_combo.currentText()
        dia = self.dia_chk.isChecked()
        mspk = self.max_spk.value()
        use_fw = (self.fw_chk.isChecked()
                  and self.deps.get("faster_whisper", False))
        self.log(f"Transcribing ({'FW' if use_fw else 'W'}) '{model}'…")
        self.tr_text.clear()
        self._busy(True)
        self.worker = TranscriptionWorker(
            self.audio_path, model, lang, dia, mspk, use_fw)
        self.worker.progress.connect(self._prog)
        self.worker.segment_ready.connect(self._seg)
        self.worker.transcript_ready.connect(self._trans_ok)
        self.worker.finished.connect(self._trans_done)
        self.worker.start()

    def _seg(self, s):
        cur = self.tr_text.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        ts = fmt_ts(s["start"])
        spk = s.get("speaker", "Speaker 1")
        fmt = QTextCharFormat()
        try:
            n = int(spk.split()[-1]) - 1
        except (ValueError, IndexError):
            n = 0
        c = SPEAKER_COLORS[n % len(SPEAKER_COLORS)]
        fmt.setForeground(QBrush(c))
        fmt.setFontWeight(QFont.Weight.Bold)
        cur.insertText(f"[{ts}] {spk}: ", fmt)
        fmt.setForeground(QBrush(QColor(0, 0, 0)))
        fmt.setFontWeight(QFont.Weight.Normal)
        cur.insertText(f"{s['text']}\n", fmt)

    def _trans_ok(self, t):
        self.transcript = t
        self._render_tr()
        self.sid_btn.setEnabled(True)
        self.ext_btn.setEnabled(True)
        self.custom_btn.setEnabled(True)

    def _trans_done(self, ok, msg):
        self._busy(False)
        self.log(f"{'✅' if ok else '❌'} {msg}")

    def _render_tr(self):
        self.tr_text.clear()
        for e in self.transcript:
            self._seg(e)

    # ────────────────── Speaker ID ──────────────────

    def _sid(self):
        if not self.transcript:
            return
        self.log("Identifying…")
        self._busy(True)
        self.worker = SpeakerIDWorker(
            self.transcript, self.video_info,
            self.ollama, self.spk_combo.currentText())
        self.worker.progress.connect(self._prog)
        self.worker.ids_ready.connect(self._ids)
        self.worker.finished.connect(lambda o, m: self._busy(False))
        self.worker.start()

    def _ids(self, ids):
        self.speaker_ids = ids
        self.sp_text.clear()
        for s, i in ids.items():
            if isinstance(i, dict):
                self.sp_text.append(
                    f"<h3>{s}</h3>"
                    f"<p><b>Identity:</b> {i.get('identity', '?')}</p>"
                    f"<p><b>Topics:</b> {', '.join(i.get('topics', []))}</p>")
        self._rerender()

    def _rerender(self):
        self.tr_text.clear()
        for e in self.transcript:
            cur = self.tr_text.textCursor()
            cur.movePosition(QTextCursor.MoveOperation.End)
            ts = fmt_ts(e["start"])
            spk = e.get("speaker", "Speaker 1")
            dn = spk
            if spk in self.speaker_ids and isinstance(
                    self.speaker_ids[spk], dict):
                ident = self.speaker_ids[spk].get("identity", "")
                if ident and ident != "Unknown":
                    dn = f"{spk} ({ident})"
            fmt = QTextCharFormat()
            try:
                n = int(spk.split()[-1]) - 1
            except (ValueError, IndexError):
                n = 0
            c = SPEAKER_COLORS[n % len(SPEAKER_COLORS)]
            fmt.setForeground(QBrush(c))
            fmt.setFontWeight(QFont.Weight.Bold)
            cur.insertText(f"[{ts}] {dn}: ", fmt)
            fmt.setForeground(QBrush(QColor(0, 0, 0)))
            fmt.setFontWeight(QFont.Weight.Normal)
            cur.insertText(f"{e['text']}\n", fmt)

    # ────────────────── Claims & Curation ──────────────────

    def _ext(self):
        if not self.transcript:
            return
        self.log("Extracting…")
        self.ctbl.setRowCount(0)
        self.claims = []
        self._busy(True)
        self.worker = ClaimExtractionWorker(
            self.transcript, self.ollama, self.ext_combo.currentText())
        self.worker.progress.connect(self._prog)
        self.worker.claim_found.connect(self._claim_row)
        self.worker.finished.connect(self._ext_done)
        self.worker.start()

    def _claim_row(self, c):
        r = self.ctbl.rowCount()
        self.ctbl.insertRow(r)
        chk = QTableWidgetItem()
        chk.setFlags(
            Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        chk.setCheckState(Qt.CheckState.Checked)
        self.ctbl.setItem(r, 0, chk)
        self.ctbl.setItem(r, 1, QTableWidgetItem(str(c.get("id", r + 1))))
        self.ctbl.setItem(r, 2, QTableWidgetItem(c.get("speaker", "?")))
        self.ctbl.setItem(r, 3, QTableWidgetItem(c.get("claim", "")[:100]))
        self.ctbl.setItem(r, 4, QTableWidgetItem(c.get("topic", "")))
        self.ctbl.setItem(r, 5, QTableWidgetItem("⏳"))

    def _ext_done(self, ok, msg, claims):
        self._busy(False)
        self.claims = claims
        self.log(f"{'✅' if ok else '❌'} {msg}")
        self.fc_btn.setEnabled(bool(claims))
        self.split_btn.setEnabled(bool(claims))
        self.custom_btn.setEnabled(True)

    def _split_claims(self):
        if not self.claims:
            return
        self.log("Splitting claims…")
        self._busy(True)
        self.worker = ClaimSplitWorker(
            self.claims, self.ollama, self.ext_combo.currentText())
        self.worker.progress.connect(self._prog)
        self.worker.split_ready.connect(self._split_done)
        self.worker.finished.connect(lambda o, m: self._busy(False))
        self.worker.start()

    def _split_done(self, new_claims):
        if not new_claims:
            self.log("No compound claims found to split.")
            return
        for c in new_claims:
            c["id"] = len(self.claims) + 1
            self.claims.append(c)
            self._claim_row(c)
        self.log(f"✅ Added {len(new_claims)} split claims. "
                 f"Uncheck originals if needed.")

    def _add_custom_claim(self):
        dlg = CustomClaimDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            c = dlg.get_claim()
            if c:
                c["id"] = len(self.claims) + 1
                self.claims.append(c)
                self._claim_row(c)
                self.fc_btn.setEnabled(True)
                self.log("✅ Custom claim added")

    def _fc(self):
        checked_claims = []
        for r in range(self.ctbl.rowCount()):
            chk = self.ctbl.item(r, 0)
            if chk and chk.checkState() == Qt.CheckState.Checked:
                cid = int(self.ctbl.item(r, 1).text())
                cl = next((c for c in self.claims if c.get("id") == cid), None)
                if cl:
                    checked_claims.append(cl)
        if not checked_claims:
            QMessageBox.warning(
                self, "Warning", "No claims checked for fact-checking.")
            return
        model = self.chk_combo.currentText()
        conc = self.conc_spin.value()
        self.log(f"Fact-checking {len(checked_claims)} checked claims "
                 f"(model: {model}, conc: {conc})…")
        self.results = []
        self.results_map = {}
        self._busy(True)
        self.worker = FactCheckWorker(
            checked_claims, self.ollama, model, max_concurrent=conc)
        self.worker.progress.connect(self._prog)
        self.worker.result_ready.connect(self._fc_res)
        self.worker.finished.connect(self._fc_done)
        self.worker.start()

    def _fc_res(self, r):
        self.results.append(r)
        cid = r.get("id", 0)
        self.results_map[cid] = r
        for row in range(self.ctbl.rowCount()):
            it = self.ctbl.item(row, 1)
            if it and it.text() == str(cid):
                v = r.get("verdict", "Unverified")
                vi = QTableWidgetItem(v)
                col = VERDICT_COLORS.get(v, QColor(149, 165, 166))
                vi.setForeground(QBrush(col))
                vi.setFont(QFont("", -1, QFont.Weight.Bold))
                self.ctbl.setItem(row, 5, vi)
                break

    def _fc_done(self, ok, msg, results):
        self._busy(False)
        self.log(f"{'✅' if ok else '❌'} {msg}")

    def _claim_click(self, row, _):
        it = self.ctbl.item(row, 1)
        if not it:
            return
        cid = int(it.text())
        if cid in self.results_map:
            self.det.setHtml(self._fmt_res(self.results_map[cid]))
        else:
            cl = next((c for c in self.claims if c.get("id") == cid), None)
            if cl:
                self.det.setHtml(
                    f"<h3>Claim #{cid}</h3>"
                    f"<p><b>Claim:</b> {cl.get('claim', '')}</p>"
                    f"<p><b>Speaker:</b> {cl.get('speaker', '?')}</p>"
                    f"<p><b>Search Query:</b> "
                    f"{cl.get('search_query', 'N/A')}</p>")

    def _fmt_res(self, r):
        v = r.get("verdict", "Unverified")
        ch = VERDICT_COLORS.get(v, QColor(149, 165, 166)).name()
        ev = "".join(f"<li>{e}</li>" for e in r.get("key_evidence", []))
        sr = "".join(f'<li><a href="{s}">{s}</a></li>'
                     for s in r.get("sources", []) if s)
        cred = r.get("source_credibility", {})
        ws = cred.get("weighted_analysis", {})
        cred_html = ""
        if ws:
            cred_html = (
                f"<p><b>Credibility:</b> "
                f"Avg {ws.get('weighted_score', 0)}/100 | "
                f"Credible: {ws.get('credible_count', 0)} | "
                f"Unreliable: {ws.get('unreliable_count', 0)}</p>")
        return (
            f"<h3>Claim #{r.get('id', '')}</h3>"
            f"<p><b>Claim:</b> {r.get('claim', '')}</p>"
            f"<p><b>Speaker:</b> {r.get('speaker', '?')}</p>"
            f"<p><b>Verdict:</b> "
            f"<span style='color:{ch};font-weight:bold'>{v}</span></p>"
            f"<p><b>Confidence:</b> {r.get('confidence', 0)}%</p>"
            f"{cred_html}"
            f"<p><b>Reasoning:</b> {r.get('reasoning', '')}</p>"
            f"<p><b>Evidence:</b></p><ul>{ev or '<li>N/A</li>'}</ul>"
            f"<p><b>Sources:</b></p><ul>{sr or '<li>N/A</li>'}</ul>")

    # ────────────────── Batch ──────────────────

    def _batch_add(self):
        urls = [u.strip() for u in self.batch_urls.toPlainText().split('\n')
                if u.strip().startswith('http')]
        for u in urls:
            if u not in self.batch_queue:
                self.batch_queue.append(u)
                r = self.batch_table.rowCount()
                self.batch_table.insertRow(r)
                self.batch_table.setItem(r, 0, QTableWidgetItem(u))
                self.batch_table.setItem(r, 1, QTableWidgetItem("Pending"))
        self.batch_urls.clear()

    def _batch_remove(self):
        rows = sorted(set(i.row() for i in self.batch_table.selectedItems()),
                      reverse=True)
        for r in rows:
            self.batch_queue.pop(r)
            self.batch_table.removeRow(r)

    def _batch_clear(self):
        self.batch_queue.clear()
        self.batch_table.setRowCount(0)

    def _batch_go(self):
        if not self.batch_queue:
            return
        self.batch_start.setEnabled(False)
        self.batch_cancel.setEnabled(True)
        self.batch_results.clear()
        self._batch_next(0)

    def _batch_next(self, idx):
        if idx >= len(self.batch_queue):
            self.batch_start.setEnabled(True)
            self.batch_cancel.setEnabled(False)
            self.batch_prog.setValue(100)
            self.log("✅ Batch done!")
            return
        url = self.batch_queue[idx]
        self.batch_prog.setValue(int((idx / len(self.batch_queue)) * 100))
        self.batch_table.setItem(idx, 1, QTableWidgetItem("Processing…"))

        use_fw = (self.fw_chk.isChecked()
                  and self.deps.get("faster_whisper", False))
        video_out = os.path.join(self.out_dir, f"video_{idx + 1}")

        self.batch_worker = BatchPipelineWorker(
            url, video_out, self.ollama,
            self.w_combo.currentText(), self.lang_combo.currentText(),
            self.dia_chk.isChecked(), self.max_spk.value(), use_fw,
            self.spk_combo.currentText(), self.ext_combo.currentText(),
            self.chk_combo.currentText())
        self._bi = idx
        self.batch_worker.progress.connect(self._batch_prog)
        self.batch_worker.status.connect(self._batch_status)
        self.batch_worker.finished.connect(self._batch_item_done)
        self.batch_worker.start()

    def _batch_prog(self, pct, msg):
        idx = getattr(self, '_bi', 0)
        total = len(self.batch_queue)
        overall = int(((idx + pct / 100) / max(total, 1)) * 100)
        self.batch_prog.setValue(overall)
        self.statusBar().showMessage(f"Batch {idx + 1}/{total}: {msg}")

    def _batch_status(self, url, status):
        if url in self.batch_queue:
            idx = self.batch_queue.index(url)
            if idx < self.batch_table.rowCount():
                self.batch_table.setItem(idx, 1, QTableWidgetItem(status))

    def _batch_item_done(self, ok, url, data):
        idx = getattr(self, '_bi', 0)
        self.batch_results[url] = data
        if idx < self.batch_table.rowCount():
            title = data.get("video_info", {}).get("title", url)[:60]
            self.batch_table.setItem(idx, 0, QTableWidgetItem(title))
            self.batch_table.setItem(
                idx, 2,
                QTableWidgetItem(str(len(data.get("claims", [])))))
            self.batch_table.setItem(
                idx, 3,
                QTableWidgetItem(str(len(data.get("results", [])))))
        self._batch_next(idx + 1)

    def _batch_stop(self):
        if self.batch_worker:
            self.batch_worker.cancel()
        self.batch_start.setEnabled(True)
        self.batch_cancel.setEnabled(False)

    def _batch_dblclick(self, row, col):
        url = self.batch_queue[row] if row < len(self.batch_queue) else ""
        if url in self.batch_results:
            data = self.batch_results[url]
            self.video_info = data.get("video_info", {})
            self.transcript = data.get("transcript", [])
            self.speaker_ids = data.get("speaker_ids", {})
            self.claims = data.get("claims", [])
            self.results = data.get("results", [])
            self.results_map = {r.get("id"): r
                                for r in self.results if "id" in r}
            self._vi(self.video_info)
            self._rerender()
            self.ctbl.setRowCount(0)
            for c in self.claims:
                self._claim_row(c)
            for r in self.results:
                self._fc_res(r)
            self.sid_btn.setEnabled(True)
            self.ext_btn.setEnabled(True)
            self.fc_btn.setEnabled(bool(self.claims))
            self.custom_btn.setEnabled(True)
            self.split_btn.setEnabled(bool(self.claims))
            self.tabs.setCurrentIndex(3)

    # ────────────────── Reporting & Export ──────────────────

    def _gen(self):
        if not self.transcript and not self.results:
            return
        vd = {}
        for r in self.results:
            v = r.get("verdict", "?")
            vd[v] = vd.get(v, 0) + 1

        spk_h = "".join(
            f'<div class="meta"><h3>{s}</h3>'
            f'<p>{i.get("identity", "?")}</p></div>'
            for s, i in self.speaker_ids.items() if isinstance(i, dict))
        tr_h = "".join(
            f'<div class="tr"><span class="ts">[{fmt_ts(e["start"])}]</span> '
            f'<span class="spk">{e.get("speaker", "?")}:</span> '
            f'{e["text"]}</div>'
            for e in self.transcript)
        fc_h = ""
        if self.results:
            fc_h += ("<table><tr><th>Verdict</th><th>Count</th></tr>"
                     + "".join(f'<tr><td>{v}</td><td>{c}</td></tr>'
                               for v, c in vd.items())
                     + "</table>")
            fc_h += (
                "<h3>Details</h3>"
                "<table><tr><th>#</th><th>Claim</th><th>Speaker</th>"
                "<th>Verdict</th><th>Conf</th><th>Cred</th></tr>"
                + "".join(
                    f'<tr><td>{r.get("id", "")}</td>'
                    f'<td>{r.get("claim", "")[:60]}</td>'
                    f'<td>{r.get("speaker", "")}</td>'
                    f'<td>{r.get("verdict", "")}</td>'
                    f'<td>{r.get("confidence", 0)}%</td>'
                    f'<td>{r.get("source_credibility", {}).get("weighted_analysis", {}).get("weighted_score", "?")}</td></tr>'
                    for r in self.results)
                + "</table>")

        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{{font-family:'Segoe UI',sans-serif;margin:40px;color:#333}}
h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
h2{{color:#2980b9}}h3{{color:#27ae60}}
.meta{{background:#ecf0f1;padding:15px;border-radius:8px;margin:15px 0}}
.tr{{background:#fafafa;padding:8px;border-left:4px solid #3498db;margin:4px 0}}
.ts{{color:#7f8c8d}}.spk{{font-weight:bold;color:#2980b9}}
table{{width:100%;border-collapse:collapse;margin:20px 0}}
th{{background:#2c3e50;color:#fff;padding:10px;text-align:left}}
td{{padding:8px;border-bottom:1px solid #ddd}}
tr:nth-child(even){{background:#f9f9f9}}
</style></head><body>
<h1>📊 Fact-Check Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="meta"><h2>Video</h2>
<p><b>Title:</b> {self.video_info.get('title', 'N/A')}</p>
<p><b>Channel:</b> {self.video_info.get('channel', 'N/A')}</p></div>
<h2>Speakers</h2>{spk_h}
<h2>Transcript</h2>{tr_h}
<h2>Fact-Check</h2>{fc_h}
<div class="footer"><p>Generated by {APP_NAME} v{APP_VERSION}</p></div>
</body></html>"""
        self.rpt.setHtml(html)
        self.log("✅ Report generated")

    def _save_html(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save HTML",
            f"report_{datetime.now().strftime('%Y%m%d')}.html",
            "HTML (*.html)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.rpt.toHtml())
            self.log(f"✅ Saved: {p}")

    def _export_csv(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            f"results_{datetime.now().strftime('%Y%m%d')}.csv",
            "CSV (*.csv)")
        if p:
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "id", "claim", "speaker", "verdict", "confidence",
                    "reasoning", "correction", "credibility_note", "sources"])
                w.writeheader()
                for r in self.results:
                    w.writerow({
                        "id": r.get("id", ""),
                        "claim": r.get("claim", ""),
                        "speaker": r.get("speaker", ""),
                        "verdict": r.get("verdict", ""),
                        "confidence": r.get("confidence", 0),
                        "reasoning": r.get("reasoning", ""),
                        "correction": r.get("correction", ""),
                        "credibility_note": r.get("credibility_note", ""),
                        "sources": "; ".join(r.get("sources", [])),
                    })
            self.log(f"✅ CSV: {p}")

    def _export_pdf(self):
        if not self.rpt.toPlainText():
            QMessageBox.warning(self, "Warning", "Generate report first")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Export PDF",
            f"report_{datetime.now().strftime('%Y%m%d')}.pdf",
            "PDF (*.pdf)")
        if p:
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(p)
            self.rpt.document().print_(printer)
            self.log(f"✅ PDF: {p}")

    # ────────────────── Session Save/Load ──────────────────

    def _save_project(self):
        if not self.transcript and not self.results:
            QMessageBox.warning(self, "Warning", "No data to save")
            return
        fname = safe_filename(self.video_info.get('title', 'project'))
        fname = fname + ".json"
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Project", fname, "Project (*.json)")
        if not p:
            return
        data = {
            "version": APP_VERSION,
            "video_info": self.video_info,
            "audio_path": self.audio_path or "",
            "speaker_ids": self.speaker_ids,
            "transcript": self.transcript,
            "claims": self.claims,
            "results": self.results,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.log(f"✅ Project saved: {p}")

    def _load_project(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Project (*.json)")
        if not p:
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.video_info = data.get("video_info", {})
        self.audio_path = data.get("audio_path", "")
        self.speaker_ids = data.get("speaker_ids", {})
        self.transcript = data.get("transcript", [])
        self.claims = data.get("claims", [])
        self.results = data.get("results", [])
        self.results_map = {r.get("id"): r
                            for r in self.results if "id" in r}
        self._vi(self.video_info)
        if self.audio_path:
            self.a_lbl.setText(f"✅ {self.audio_path}")
        self._rerender()
        self.ctbl.setRowCount(0)
        for c in self.claims:
            self._claim_row(c)
        for r in self.results:
            self._fc_res(r)
        self.sid_btn.setEnabled(bool(self.transcript))
        self.ext_btn.setEnabled(bool(self.transcript))
        self.fc_btn.setEnabled(bool(self.claims))
        self.custom_btn.setEnabled(True)
        self.split_btn.setEnabled(bool(self.claims))
        self.log(f"✅ Project loaded: {p}")

    # ────────────────── Contradiction Matrix ──────────────────

    def _contradiction_matrix(self):
        if not self.results:
            QMessageBox.warning(
                self, "Warning", "Fact-check results needed first")
            return
        self.log("Analyzing contradictions…")
        self._busy(True)
        self.worker = ContradictionWorker(
            self.results, self.ollama, self.chk_combo.currentText())
        self.worker.progress.connect(self._prog)
        self.worker.contradiction_ready.connect(self._cont_done)
        self.worker.finished.connect(lambda o, m: self._busy(False))
        self.worker.start()

    def _cont_done(self, html):
        dlg = QDialog(self)
        dlg.setWindowTitle("⚡ Speaker Contradiction Matrix")
        dlg.setMinimumSize(800, 500)
        lay = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setHtml(html)
        lay.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.exec()

    # ────────────────── UI Helpers ──────────────────

    def _busy(self, on):
        self.pbar.setVisible(on)
        self.pbar.setValue(0 if on else 100)
        self.cn_btn.setEnabled(on)
        self.fc_cn.setEnabled(on)
        for b in (self.tr_btn, self.sid_btn, self.ext_btn,
                  self.fc_btn, self.split_btn, self.custom_btn):
            b.setEnabled(not on)

    def _prog(self, pct, msg):
        self.pbar.setValue(pct)
        self.statusBar().showMessage(msg)

    def _cancel(self):
        if self.worker and hasattr(self.worker, 'cancel'):
            self.worker.cancel()
        self._busy(False)

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_.append(f"[{ts}] {msg}")
        self.log_.verticalScrollBar().setValue(
            self.log_.verticalScrollBar().maximum())
        logger.info(msg)


# ═══════════════════════ Theme ═══════════════════════

def dark_theme(app):
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(35, 35, 35))
    app.setPalette(p)
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold; border: 1px solid #555;
            border-radius: 6px; margin-top: 12px; padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 10px; padding: 0 5px;
        }
        QPushButton {
            padding: 6px 14px; border-radius: 4px;
            border: 1px solid #666; background: #444;
        }
        QPushButton:hover { background: #555; }
        QPushButton:pressed { background: #333; }
        QPushButton:disabled { background: #333; color: #666; }
        QTabWidget::pane { border: 1px solid #555; border-radius: 4px; }
        QTabBar::tab {
            padding: 8px 16px; border: 1px solid #555;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px; margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #444; border-bottom: 2px solid #2a82da;
        }
        QTabBar::tab:!selected { background: #333; }
        QTableWidget { gridline-color: #555; }
        QTableWidget::item:selected { background: #2a82da; }
        QProgressBar {
            border: 1px solid #555; border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background: #2a82da; border-radius: 3px;
        }
        QTextEdit, QLineEdit, QComboBox {
            border: 1px solid #555; border-radius: 4px; padding: 4px;
        }
        QScrollBar:vertical {
            border: none; background: #353535; width: 10px;
        }
        QScrollBar::handle:vertical {
            background: #666; border-radius: 5px; min-height: 20px;
        }
        QRadioButton, QCheckBox { spacing: 8px; }
    """)

# ═══════════════════════ Main ═══════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    dark_theme(app)
    w = MainWindow()
    w.show()
    QTimer.singleShot(500, w._chk_ollama)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()