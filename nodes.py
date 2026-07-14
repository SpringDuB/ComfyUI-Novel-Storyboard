from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import wave
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

try:
    import folder_paths
except ImportError:  # Allows lightweight unit tests outside a ComfyUI install.
    folder_paths = None


MAX_SEED = 2**53 - 1
S2V_FPS = 16
S2V_MIN_FRAMES = 73
S2V_MAX_FRAMES = 97
S2V_DEFAULT_MAX_DIALOGUE_SECONDS = 15.0
VOICE_SLOTS = tuple("ABCDEFGH")


def _first(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("The LLM response did not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


def _chat_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


def _read_streamed_chat_content(response: Any) -> str:
    content_parts: list[str] = []
    event_data: list[str] = []
    plain_body: list[str] = []
    saw_sse = False
    finished = False

    def consume_event() -> bool:
        nonlocal event_data
        if not event_data:
            return False
        payload = "\n".join(event_data).strip()
        event_data = []
        if not payload:
            return False
        if payload == "[DONE]":
            return True
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Storyboard API returned malformed SSE JSON: {payload[:1000]}") from error
        if isinstance(event, dict) and event.get("error"):
            raise RuntimeError(f"Storyboard API stream returned an error: {str(event['error'])[:1000]}")
        try:
            delta = event["choices"][0].get("delta", {})
        except (KeyError, IndexError, TypeError, AttributeError):
            return False
        if isinstance(delta, dict):
            text = _chat_content_text(delta.get("content"))
            if text:
                content_parts.append(text)
        return False

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        plain_body.append(line)
        line = line.rstrip("\r\n")
        if not line:
            if consume_event():
                finished = True
                break
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            saw_sse = True
            event_data.append(line[5:].lstrip())

    if not finished:
        consume_event()
    if saw_sse:
        if not content_parts:
            raise RuntimeError("Storyboard API stream ended without choices[0].delta.content.")
        return "".join(content_parts)

    try:
        response_body = json.loads("".join(plain_body))
        return _chat_content_text(response_body["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "Storyboard API returned neither valid SSE data nor Chat Completions JSON: "
            f"{''.join(plain_body)[:1000]}"
        ) from error


def _find_storyboard_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    shots = value.get("shots")
    if isinstance(shots, list) and any(
        isinstance(shot, dict)
        and str(shot.get("image_prompt", "")).strip()
        and str(shot.get("video_prompt", "")).strip()
        for shot in shots
    ):
        return value
    for key in (
        "revised_storyboard",
        "storyboard",
        "result",
        "data",
        "output",
        "response",
        "draft_storyboard",
    ):
        candidate = _find_storyboard_payload(value.get(key))
        if candidate is not None:
            return candidate
    return None


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:7], "big")
    return int((base_seed + offset) % MAX_SEED)


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,，、|]", value) if item.strip()]
    return []


def _s2v_frame_count(duration: float) -> int:
    requested = max(S2V_MIN_FRAMES, min(S2V_MAX_FRAMES, round(duration * S2V_FPS)))
    return int(1 + 4 * round((requested - 1) / 4))


def _assign_voice_slots(
    characters: dict[str, dict[str, Any]],
    raw_shots: list[Any],
) -> dict[str, str]:
    speaking_order: list[str] = []
    for raw in raw_shots:
        if not isinstance(raw, dict):
            continue
        dialogue = str(raw.get("dialogue_text", "")).strip()
        speaker_id = str(raw.get("speaker_id", "")).strip()
        if dialogue and not speaker_id:
            candidates = [
                str(raw.get("primary_character_id", "")).strip(),
                *_as_string_list(raw.get("characters")),
            ]
            speaker_id = next((candidate for candidate in candidates if candidate), "")
            if speaker_id:
                raw["speaker_id"] = speaker_id
        if dialogue and speaker_id and speaker_id not in speaking_order:
            speaking_order.append(speaker_id)
        if speaker_id and speaker_id not in characters:
            characters[speaker_id] = {
                "id": speaker_id,
                "name": speaker_id,
                "gender": "unknown",
                "identity_anchor": "",
                "voice_profile": "male",
                "voice_identity": "成年普通话声线，音色自然，语速适中",
            }

    assigned: dict[str, str] = {}
    used: set[str] = set()
    for character_id in speaking_order:
        requested = str(characters[character_id].get("voice_slot", "")).strip().upper()
        if requested in VOICE_SLOTS and requested not in used:
            assigned[character_id] = requested
            used.add(requested)

    for character_id in speaking_order:
        if character_id in assigned:
            continue
        available = next((slot for slot in VOICE_SLOTS if slot not in used), None)
        if available is None:
            raise ValueError(
                "A single workflow supports up to eight speaking characters. "
                "Split the chapter or remove dialogue from minor roles beyond voice slot H."
            )
        assigned[character_id] = available
        used.add(available)

    for character_id, character in characters.items():
        character["voice_slot"] = assigned.get(character_id, "")
    return assigned


def _normalize_storyboard(
    payload: dict[str, Any],
    *,
    max_shots: int,
    seconds_per_shot: float,
    style_prompt: str,
    avoid_prompt: str,
    base_seed: int,
) -> dict[str, Any]:
    raw_shots = payload.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise ValueError("The LLM response must contain a non-empty 'shots' array.")

    continuity_bible = payload.get("continuity_bible", {})
    if not isinstance(continuity_bible, dict):
        continuity_bible = {}
    raw_characters = continuity_bible.get("characters", [])
    characters = {}
    if isinstance(raw_characters, list):
        for index, character in enumerate(raw_characters):
            if not isinstance(character, dict):
                continue
            character_id = str(character.get("id") or f"C{index + 1:02d}").strip()
            name = str(character.get("name") or character_id).strip()
            identity_anchor = str(
                character.get("identity_anchor")
                or character.get("immutable_visual_description")
                or ""
            ).strip()
            gender = str(character.get("gender", "unknown")).strip().lower()
            voice_profile = str(character.get("voice_profile", "")).strip().lower()
            if voice_profile not in {"male", "female", "narrator"}:
                voice_profile = "female" if gender in {"female", "woman", "女"} else "male"
            voice_identity = str(character.get("voice_identity", "")).strip()
            if not voice_identity:
                voice_identity = (
                    "成年女性普通话声线，音色自然，音高中等，语速适中"
                    if voice_profile == "female"
                    else "成年男性普通话声线，音色自然，音高中低，语速适中"
                )
            characters[character_id] = {
                **character,
                "id": character_id,
                "name": name,
                "identity_anchor": identity_anchor,
                "voice_profile": voice_profile,
                "voice_identity": voice_identity,
            }
    voice_slots = _assign_voice_slots(characters, raw_shots)
    if characters:
        continuity_bible["characters"] = list(characters.values())

    shots = []
    for index, raw in enumerate(raw_shots[:max_shots]):
        if not isinstance(raw, dict):
            continue

        image_prompt = str(raw.get("image_prompt", "")).strip()
        video_prompt = str(raw.get("video_prompt", "")).strip()
        if not image_prompt or not video_prompt:
            continue

        character_ids = _as_string_list(raw.get("characters"))
        speaker_id = str(raw.get("speaker_id", "")).strip()
        if speaker_id and speaker_id not in character_ids:
            character_ids.append(speaker_id)
        primary_character_id = str(raw.get("primary_character_id", "")).strip()
        if primary_character_id not in character_ids:
            primary_character_id = character_ids[0] if character_ids else ""

        identity_lines = []
        for character_id in character_ids:
            character = characters.get(character_id)
            if not character:
                continue
            anchor = character.get("identity_anchor", "")
            if anchor:
                identity_lines.append(
                    f"{character_id}「{character['name']}」固定身份：{anchor}"
                )
        if identity_lines:
            image_prompt = (
                f"【角色身份锁定，不得改变脸型、五官比例、发型和年龄】{'；'.join(identity_lines)}。"
                f"【本镜画面】{image_prompt}"
            )

        continuity = str(raw.get("continuity_from_previous", "")).strip()
        if continuity and index > 0:
            image_prompt = f"【承接上一镜】{continuity}。【连续性要求】保持轴线、视线方向、服装、道具位置、时间和光线一致。{image_prompt}"
        if style_prompt and style_prompt not in image_prompt:
            image_prompt = f"{image_prompt}。【统一视觉风格】{style_prompt}"
        if avoid_prompt:
            image_prompt = f"{image_prompt}。【排除】{avoid_prompt}"

        dialogue_text = str(raw.get("dialogue_text", "")).strip().strip("“”\"'")
        narration_text = str(
            raw.get("narration_text") or raw.get("narration_or_dialogue") or ""
        ).strip()
        audio_type = str(raw.get("audio_type", "")).strip().lower()
        if dialogue_text:
            audio_type = "dialogue"
        elif audio_type not in {"dialogue", "narration"}:
            audio_type = "narration" if narration_text else "none"

        speech_framing = str(raw.get("speech_framing", "none")).strip().lower()
        if audio_type == "dialogue":
            if speech_framing not in {"group", "closeup"}:
                speech_framing = "closeup" if len(dialogue_text) >= 10 else "group"
            speaker = characters.get(speaker_id, {})
            speaker_name = speaker.get("name", speaker_id or "发言者")
            if speech_framing == "group":
                speech_instruction = (
                    f"群像对话镜头：只有{speaker_name}说话，使用普通话清晰说出“{dialogue_text}”，"
                    "嘴唇、下颌和面部肌肉严格跟随真实语音音素自然运动；发言者口部无遮挡且清晰可见。"
                    "其他人物保持闭嘴，只做自然倾听、视线和轻微表情反应，禁止多人同时动嘴。"
                )
            else:
                speech_instruction = (
                    f"切到{speaker_name}的近景或特写，正面或轻微四分之三侧面，口部无遮挡；"
                    f"{speaker_name}使用普通话清晰说出“{dialogue_text}”，嘴唇、下颌和面部肌肉严格跟随"
                    "真实语音音素逐字自然运动，保持同一张脸、同一发型和同一服装。其他人物不抢镜。"
                )
            video_prompt = f"{video_prompt}。{speech_instruction}"
            tts_text = dialogue_text
            speaker = characters.get(speaker_id, {})
            voice_profile = speaker.get("voice_profile", "male")
            voice_slot = voice_slots.get(speaker_id, "")
            voice_identity = speaker.get("voice_identity", "成年普通话声线，音色自然，语速适中")
            performance = str(
                raw.get("voice_instruction")
                or raw.get("speaking_style")
                or "贴合当前剧情和人物情绪，自然克制地说话"
            ).strip()
            if not re.search(r"[\u3400-\u9fff]", performance):
                performance = "贴合当前剧情和人物情绪，自然克制地说话"
            voice_instruction = (
                f"严格保持参考音频中的同一人物声线身份，固定为：{voice_identity}。"
                "不得改变年龄感、基础音色、音高范围、口音和基础说话节奏；"
                f"本句只调整表演状态：{performance}。使用自然清晰的普通话，避免播音腔和夸张表演。"
            )
            voice_seed = _stable_seed(base_seed, f"voice:{speaker_id}")
        else:
            speech_framing = "none"
            video_prompt = f"{video_prompt}。所有可见人物保持闭嘴，不做说话口型，只做符合情境的自然呼吸和表情变化。"
            tts_text = ""
            voice_profile = "narrator"
            voice_slot = ""
            voice_instruction = ""
            voice_seed = _stable_seed(base_seed, "voice:silence")

        video_prompt = (
            f"{video_prompt}。严格保持起始图中的人物身份、五官、发型、服装、场景布局、色调和光线不变；"
            "动作连续、物理合理，不新增人物，不瞬移，不改变镜头轴线。"
        )

        try:
            duration = float(raw.get("duration_seconds", seconds_per_shot))
        except (TypeError, ValueError):
            duration = seconds_per_shot
        duration = round(max(4.5, min(6.0, duration)), 2)
        frame_count = _s2v_frame_count(duration)
        duration = round(frame_count / S2V_FPS, 4)

        scene_id = str(raw.get("scene_id") or "SC001").strip()
        sequence_id = str(raw.get("sequence_id") or scene_id).strip()
        transition = str(raw.get("transition", "hard_cut")).strip().lower()
        transition_aliases = {
            "cut": "hard_cut",
            "硬切": "hard_cut",
            "直接切": "hard_cut",
            "dissolve": "dissolve",
            "叠化": "dissolve",
            "淡入淡出": "dissolve",
            "match": "match_cut",
            "匹配剪辑": "match_cut",
        }
        transition = transition_aliases.get(transition, transition)
        if transition not in {"hard_cut", "dissolve", "match_cut"}:
            transition = "hard_cut"

        seed_key = f"character:{primary_character_id}" if primary_character_id else f"scene:{scene_id}"

        shot_id = str(raw.get("id") or f"S{index + 1:03d}").strip()
        shots.append(
            {
                "id": shot_id,
                "title": str(raw.get("title", f"Shot {index + 1}")).strip(),
                "sequence_id": sequence_id,
                "scene_id": scene_id,
                "characters": character_ids,
                "primary_character_id": primary_character_id,
                "continuity_from_previous": continuity,
                "transition": transition,
                "duration_seconds": duration,
                "frame_count": frame_count,
                "image_prompt": image_prompt,
                "video_prompt": video_prompt,
                "audio_type": audio_type,
                "speaker_id": speaker_id,
                "speech_framing": speech_framing,
                "dialogue_text": dialogue_text,
                "narration_text": narration_text,
                "tts_text": tts_text,
                "voice_profile": voice_profile,
                "voice_slot": voice_slot,
                "voice_instruction": voice_instruction,
                "voice_seed": voice_seed,
                "seed_group": seed_key,
                "seed": _stable_seed(base_seed, seed_key),
            }
        )

    if not shots:
        raise ValueError("No valid shots with both image_prompt and video_prompt were returned.")

    return {
        "project": payload.get("project", {}),
        "continuity_bible": continuity_bible,
        "shots": shots,
    }


def _transition_overlap(
    index: int,
    transitions: list[str] | None,
    crossfade_frames: int,
) -> int:
    if not transitions or index >= len(transitions):
        return max(0, crossfade_frames)
    transition = str(transitions[index]).strip().lower()
    return max(0, crossfade_frames) if transition == "dissolve" else 0


def _concat_frame_batches(
    frame_batches: list[torch.Tensor],
    crossfade_frames: int,
    transitions: list[str] | None = None,
) -> torch.Tensor:
    if not frame_batches:
        raise ValueError("At least one frame batch is required.")

    result = frame_batches[0]
    if result.ndim != 4:
        raise ValueError("Video frames must use ComfyUI's [frames, height, width, channels] shape.")

    for index, next_frames in enumerate(frame_batches[1:], start=1):
        if next_frames.ndim != 4:
            raise ValueError("Video frames must use ComfyUI's [frames, height, width, channels] shape.")
        if result.shape[1:] != next_frames.shape[1:]:
            raise ValueError(
                f"All clips must have the same frame size. Got {tuple(result.shape[1:])} "
                f"and {tuple(next_frames.shape[1:])}."
            )

        overlap = min(
            _transition_overlap(index, transitions, crossfade_frames),
            result.shape[0],
            next_frames.shape[0],
        )
        if overlap == 0:
            result = torch.cat((result, next_frames), dim=0)
            continue

        weights = torch.linspace(
            0.0,
            1.0,
            overlap,
            dtype=result.dtype,
            device=result.device,
        ).view(overlap, 1, 1, 1)
        blended = result[-overlap:] * (1.0 - weights) + next_frames[:overlap] * weights
        result = torch.cat((result[:-overlap], blended, next_frames[overlap:]), dim=0)

    return result


def _silence_audio(duration: float, sample_rate: int = 24000) -> dict[str, Any]:
    samples = max(1, round(duration * sample_rate))
    return {
        "waveform": torch.zeros((1, 1, samples), dtype=torch.float32),
        "sample_rate": sample_rate,
    }


def _fit_audio_duration(audio: dict[str, Any], duration: float) -> dict[str, Any]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 1:
        waveform = waveform.view(1, 1, -1)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    target_samples = max(1, round(duration * sample_rate))
    if waveform.shape[-1] > target_samples:
        waveform = waveform[..., :target_samples]
    elif waveform.shape[-1] < target_samples:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
    return {"waveform": waveform.contiguous(), "sample_rate": sample_rate}


def _decode_wav_bytes(data: bytes) -> dict[str, Any]:
    if not data.startswith(b"RIFF"):
        raise ValueError("TTS service did not return a WAV file.")
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        samples = torch.frombuffer(bytearray(raw), dtype=torch.uint8).float()
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    elif sample_width == 3:
        bytes_tensor = torch.tensor(list(raw), dtype=torch.int32).view(-1, 3)
        samples = (
            bytes_tensor[:, 0]
            | (bytes_tensor[:, 1] << 8)
            | (bytes_tensor[:, 2] << 16)
        )
        samples = torch.where(samples >= 2**23, samples - 2**24, samples).float() / float(2**23)
    elif sample_width == 4:
        samples = torch.frombuffer(bytearray(raw), dtype=torch.int32).float() / float(2**31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")

    if samples.numel() % channels != 0:
        raise ValueError("Invalid interleaved WAV sample count.")
    waveform = samples.view(-1, channels).transpose(0, 1).unsqueeze(0).contiguous()
    return {"waveform": waveform, "sample_rate": sample_rate}


def _concat_audio_batches(
    audio_batches: list[dict[str, Any] | None],
    frame_counts: list[int],
    fps: float,
    transitions: list[str] | None,
    crossfade_frames: int,
) -> dict[str, Any] | None:
    first_audio = next((audio for audio in audio_batches if audio is not None), None)
    if first_audio is None:
        return None

    sample_rate = int(first_audio["sample_rate"])
    fitted = []
    for audio, frames in zip(audio_batches, frame_counts):
        duration = frames / fps
        if audio is None:
            fitted.append(_silence_audio(duration, sample_rate)["waveform"])
            continue
        waveform = audio["waveform"]
        if int(audio["sample_rate"]) != sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(
                waveform,
                int(audio["sample_rate"]),
                sample_rate,
            )
        fitted.append(_fit_audio_duration(
            {"waveform": waveform, "sample_rate": sample_rate},
            duration,
        )["waveform"])

    result = fitted[0]
    for index, next_audio in enumerate(fitted[1:], start=1):
        overlap_frames = _transition_overlap(index, transitions, crossfade_frames)
        overlap_samples = min(
            round(overlap_frames / fps * sample_rate),
            result.shape[-1],
            next_audio.shape[-1],
        )
        if overlap_samples <= 0:
            result = torch.cat((result, next_audio), dim=-1)
            continue
        weights = torch.linspace(
            0.0,
            1.0,
            overlap_samples,
            dtype=result.dtype,
            device=result.device,
        ).view(1, 1, overlap_samples)
        blended = result[..., -overlap_samples:] * (1.0 - weights) + next_audio[..., :overlap_samples] * weights
        result = torch.cat((result[..., :-overlap_samples], blended, next_audio[..., overlap_samples:]), dim=-1)

    return {"waveform": result.contiguous(), "sample_rate": sample_rate}


class NWFNovelChapterPlanner:
    CATEGORY = "Novel Workflow"
    FUNCTION = "plan"
    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "image_prompts",
        "video_prompts",
        "shot_ids",
        "seeds",
        "durations",
        "frame_counts",
        "tts_texts",
        "voice_profiles",
        "voice_slots",
        "voice_instructions",
        "voice_seeds",
        "transitions",
        "storyboard_json",
    )
    OUTPUT_IS_LIST = (
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chapter": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "在这里粘贴小说的单个章节……",
                    },
                ),
                "api_base": (
                    "STRING",
                    {"default": "http://127.0.0.1:11434/v1"},
                ),
                "api_key_env": (
                    "STRING",
                    {"default": "OPENAI_API_KEY"},
                ),
                "model": (
                    "STRING",
                    {"default": "qwen3:8b"},
                ),
                "max_shots": (
                    "INT",
                    {"default": 12, "min": 1, "max": 60, "step": 1},
                ),
                "seconds_per_shot": (
                    "FLOAT",
                    {"default": 5.0, "min": 4.5, "max": 6.0, "step": 0.25},
                ),
                "style_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "电影级中国真人影视质感，真实皮肤与织物纹理，精细美术置景，"
                            "自然且有方向性的光线，16:9电影画幅，统一电影调色，细节清晰"
                        ),
                    },
                ),
                "avoid_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "文字，字幕，水印，标志，重复人物，畸形手部，多余手指，脸部漂移，"
                            "服装不一致，年龄变化，低清晰度，过度饱和，塑料皮肤"
                        ),
                    },
                ),
                "continuity_review": ("BOOLEAN", {"default": True}),
                "base_seed": (
                    "INT",
                    {"default": 20260713, "min": 0, "max": MAX_SEED, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05},
                ),
                "timeout_seconds": (
                    "INT",
                    {"default": 300, "min": 30, "max": 1800, "step": 30},
                ),
            }
        }

    @staticmethod
    def _chat_completion(
        api_base: str,
        api_key_env: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        endpoint = api_base.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "max_tokens": 16000,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        api_key = os.environ.get(api_key_env.strip(), "") if api_key_env.strip() else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                content = _read_streamed_chat_content(response)
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Storyboard API returned HTTP {error.code}: {details[:1000]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not reach the storyboard API at {endpoint}: {error.reason}"
            ) from error

        try:
            return _extract_json(str(content))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "The streamed Chat Completions content did not contain a parseable storyboard JSON: "
                f"{str(content)[:1000]}"
            ) from error

    @classmethod
    def _request_storyboard(
        cls,
        chapter: str,
        api_base: str,
        api_key_env: str,
        model: str,
        max_shots: int,
        seconds_per_shot: float,
        style_prompt: str,
        temperature: float,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        system_prompt = f"""
你是一名中国影视导演、分镜师和场记。请把一个完整的中文小说章节转换为可直接用于
Z-Image-Turbo 文生图与 Wan 2.2 S2V 音频驱动视频的制作级分镜。

只返回一个合法 JSON 对象，不要 Markdown，不要解释。JSON 键名保持下面的英文，所有用于模型的
提示词和所有描述值必须使用简体中文：
{{
  "project": {{"title": "...", "summary": "..."}},
  "continuity_bible": {{
    "visual_style": "...",
    "characters": [{{
      "id": "C01", "name": "...", "gender": "male或female",
      "identity_anchor": "不可变的中文身份描述：年龄、脸型、五官比例、肤色、发型发色、体型、辨识特征",
      "default_wardrobe": "...", "voice_profile": "male或female",
      "voice_slot": "A到H中的唯一字母，仅有台词的角色需要",
      "voice_identity": "不可变的中文声线档案：年龄感、音色厚薄、音高范围、口音、咬字、基础语速与节奏"
    }}],
    "locations": [{{"id": "L01", "name": "...", "identity_anchor": "空间布局、材质、主光方向和固定道具"}}]
  }},
  "shots": [
    {{
      "id": "S001",
      "sequence_id": "SEQ01",
      "scene_id": "SC001",
      "title": "...",
      "duration_seconds": {seconds_per_shot},
      "characters": ["C01"],
      "primary_character_id": "C01",
      "continuity_from_previous": "上一镜末尾的人物位置、视线、动作、道具、光线如何接到本镜；首镜为空",
      "transition": "hard_cut或dissolve或match_cut",
      "image_prompt": "给Z-Image-Turbo的完整中文提示词",
      "video_prompt": "给Wan 2.2的完整中文动作与运镜提示词",
      "audio_type": "dialogue或narration或none",
      "speaker_id": "C01或空",
      "speech_framing": "group或closeup或none",
      "dialogue_text": "需要角色当场说出的简短中文台词或空字符串",
      "voice_instruction": "本句的情绪强度、气息、音量和表演状态；不得改写角色的固定声线身份",
      "narration_text": "画外旁白或空字符串"
    }}
  ]
}}

硬性规则：
1. 最多 {max_shots} 镜，按章节时间顺序覆盖开端、发展、转折和结尾。一个场景尽量连续安排2-4镜，
   不要把每句文字机械切成一镜。
2. 先建立严格的角色与场景连续性圣经。identity_anchor 必须具体、可视、固定；同一人物绝不改变
   年龄、脸型、五官比例、发型发色和辨识特征。characters 按画面叙事重要性排序，并明确
   primary_character_id，连续镜头尽量保持同一个主身份角色。所有有台词角色还必须建立不可变的
   voice_identity，并按首次发言顺序分配唯一 voice_slot A-H；同一角色全章不得改变槽位或声线档案。
3. 每条 image_prompt 都必须是独立完整的中文提示词，并逐字重复所有可见角色的 identity_anchor。
   必须明确：主体和动作、人物当下状态与微表情、环境与前中后景、景别、机位、镜头角度、构图法、
   主光方向与光质、辅光/轮廓光、色温、综合色调、景深、材质细节、氛围，以及烟尘、雨雪、火花、
   能量、体积光等实际存在的特效。统一风格为：{style_prompt}
4. 连续场景必须遵守180度轴线、人物屏幕方向、视线匹配、动作匹配、服装、伤痕、道具位置、天气、
   时间和光线连续。continuity_from_previous 要写成可执行的视觉承接，不要只写“承接上一镜”。
5. video_prompt 必须使用中文，只描述起始图上可发生的主体动作、环境运动、特效变化、运镜、节奏和
   必须保持不变的内容；不得凭空新增人物或场景，不得瞬移。
6. 对白镜头必须填写 dialogue_text、speaker_id 和 speech_framing：
   - group：对话关系和其他人的反应重要时用群像/双人中景，明确只有发言者动嘴，其他人闭嘴倾听。
   - closeup：情绪爆发、关键信息或台词较长时切到发言者近景/特写，正面或轻微四分之三角度，
     口部无遮挡。不要连续滥用特写；先有建立镜头再切近景。
   每个对白镜头的台词应能在约 {seconds_per_shot} 秒内自然说完；过长台词拆成多个连续镜头。
   voice_instruction 只描述本句情绪、气息、音量与表演强度，必须要求保持参考音色、年龄感、口音
   和基础节奏，不得用“换成另一种声线”之类会改变角色身份的指令。
7. 旁白与对白严格分开。旁白不让画面人物动嘴。
8. 场景内优先 hard_cut 或 match_cut；只有明显时间/地点变化才用 dissolve。
9. 保留原文事实，不虚构重大剧情，不要求模型生成字幕、对白文字、水印或标志。
""".strip()
        return cls._chat_completion(
            api_base=api_base,
            api_key_env=api_key_env,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chapter},
            ],
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def _review_storyboard(
        cls,
        draft: dict[str, Any],
        chapter: str,
        api_base: str,
        api_key_env: str,
        model: str,
        temperature: float,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        review_prompt = """
你是影视剧组的总场记和对白镜头审核员。检查草稿中的每一镜，并直接返回修订后的完整 JSON。
只返回 JSON，不要解释。顶级必须原样保留 project、continuity_bible、shots 三个字段；shots 必须是包含完整分镜的
非空数组。禁止只返回审核意见，禁止返回空 shots，禁止套入 revised_storyboard、result 或 data 等外层字段。必须修复：
- 角色 identity_anchor 是否在每条相关 image_prompt 中逐字一致；
- 相邻镜头的轴线、屏幕方向、视线、动作、服装、伤痕、道具、天气、时间、光线和色调是否连续；
- 构图、光线、色调、人物状态、景深、材质和特效是否写全；
- 是否存在无意义跳切、重复镜头或地点突然变化；
- 对白是否正确区分 group 群像说话与 closeup 本人近景说话；发言者必须唯一，口部可见，其他人闭嘴；
- 每个有台词角色是否具有唯一且全章不变的 voice_slot A-H 和具体 voice_identity；每镜 voice_instruction
  是否只改变情绪表演而不改变音色、年龄感、口音与基础节奏；
- 旁白镜头不得让人物对口型；
- 台词是否短到能在单镜时长内自然说完，过长则拆分连续镜头；
- 所有 image_prompt 和 video_prompt 必须使用简体中文。
不得改变小说的主要事实、人物关系和事件顺序。
""".strip()
        user_content = json.dumps(
            {"source_chapter": chapter, "draft_storyboard": draft},
            ensure_ascii=False,
        )
        reviewed_raw = cls._chat_completion(
            api_base=api_base,
            api_key_env=api_key_env,
            model=model,
            messages=[
                {"role": "system", "content": review_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=min(temperature, 0.25),
            timeout_seconds=timeout_seconds,
        )
        reviewed = _find_storyboard_payload(reviewed_raw)
        draft_storyboard = _find_storyboard_payload(draft)
        if draft_storyboard is None:
            raise ValueError("The initial storyboard draft did not contain any usable shots.")
        if reviewed is None:
            keys = ", ".join(str(key) for key in reviewed_raw.keys()) or "<none>"
            print(
                "[NWFNovelChapterPlanner] WARNING: continuity review returned no usable shots "
                f"(top-level keys: {keys}); keeping the valid initial draft."
            )
            return draft_storyboard

        merged = {**draft_storyboard, **reviewed}
        if not isinstance(reviewed.get("project"), dict):
            merged["project"] = draft_storyboard.get("project", {})
        if not isinstance(reviewed.get("continuity_bible"), dict):
            merged["continuity_bible"] = draft_storyboard.get("continuity_bible", {})
        return merged

    def plan(
        self,
        chapter,
        api_base,
        api_key_env,
        model,
        max_shots,
        seconds_per_shot,
        style_prompt,
        avoid_prompt,
        continuity_review,
        base_seed,
        temperature,
        timeout_seconds,
    ):
        chapter = chapter.strip()
        if len(chapter) < 20:
            raise ValueError("Chapter text is too short. Paste one complete novel chapter.")

        payload = self._request_storyboard(
            chapter=chapter,
            api_base=api_base,
            api_key_env=api_key_env,
            model=model,
            max_shots=max_shots,
            seconds_per_shot=seconds_per_shot,
            style_prompt=style_prompt.strip(),
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        payload = _find_storyboard_payload(payload) or payload
        if continuity_review:
            payload = self._review_storyboard(
                draft=payload,
                chapter=chapter,
                api_base=api_base,
                api_key_env=api_key_env,
                model=model,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
        storyboard = _normalize_storyboard(
            payload,
            max_shots=max_shots,
            seconds_per_shot=seconds_per_shot,
            style_prompt=style_prompt.strip(),
            avoid_prompt=avoid_prompt.strip(),
            base_seed=base_seed,
        )
        storyboard["metadata"] = {
            "generator": "ComfyUI-Novel-Storyboard",
            "model": model,
            "source_character_count": len(chapter),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        shots = storyboard["shots"]
        return (
            [shot["image_prompt"] for shot in shots],
            [shot["video_prompt"] for shot in shots],
            [shot["id"] for shot in shots],
            [shot["seed"] for shot in shots],
            [shot["duration_seconds"] for shot in shots],
            [shot["frame_count"] for shot in shots],
            [shot["tts_text"] for shot in shots],
            [shot["voice_profile"] for shot in shots],
            [shot["voice_slot"] for shot in shots],
            [shot["voice_instruction"] for shot in shots],
            [shot["voice_seed"] for shot in shots],
            [shot["transition"] for shot in shots],
            json.dumps(storyboard, ensure_ascii=False, indent=2),
        )


class NWFTextToSpeech:
    CATEGORY = "Novel Workflow"
    FUNCTION = "synthesize"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "voice_profile": ("STRING", {"forceInput": True}),
                "target_duration": ("FLOAT", {"forceInput": True}),
                "api_base": ("STRING", {"default": "http://127.0.0.1:8880/v1"}),
                "api_key_env": ("STRING", {"default": "TTS_API_KEY"}),
                "model": ("STRING", {"default": "kokoro"}),
                "male_voice": ("STRING", {"default": "zm_yunxi"}),
                "female_voice": ("STRING", {"default": "zf_xiaobei"}),
                "narrator_voice": ("STRING", {"default": "zf_xiaoxiao"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "silence_sample_rate": (
                    "INT",
                    {"default": 24000, "min": 8000, "max": 48000, "step": 1000},
                ),
                "max_dialogue_seconds": (
                    "FLOAT",
                    {
                        "default": S2V_DEFAULT_MAX_DIALOGUE_SECONDS,
                        "min": 6.0,
                        "max": 30.0,
                        "step": 0.5,
                    },
                ),
            }
        }

    def synthesize(
        self,
        text,
        voice_profile,
        target_duration,
        api_base,
        api_key_env,
        model,
        male_voice,
        female_voice,
        narrator_voice,
        speed,
        silence_sample_rate,
    ):
        target_duration = float(target_duration)
        text = str(text).strip()
        if not text:
            return (_silence_audio(target_duration, int(silence_sample_rate)),)

        profile = str(voice_profile).strip().lower()
        if profile == "female":
            voice = female_voice
        elif profile == "narrator":
            voice = narrator_voice
        else:
            voice = male_voice

        endpoint = api_base.rstrip("/")
        if not endpoint.endswith("/audio/speech"):
            endpoint += "/audio/speech"
        body = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": float(speed),
        }
        headers = {"Content-Type": "application/json", "Accept": "audio/wav"}
        api_key = os.environ.get(api_key_env.strip(), "") if api_key_env.strip() else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                audio_bytes = response.read()
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS API returned HTTP {error.code}: {details[:1000]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach the TTS API at {endpoint}: {error.reason}") from error

        try:
            audio = _decode_wav_bytes(audio_bytes)
        except ValueError as error:
            preview = audio_bytes[:500].decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS API returned invalid WAV data: {preview}") from error
        return (_fit_audio_duration(audio, target_duration),)


class NWFSelectVoiceReference:
    CATEGORY = "Novel Workflow"
    FUNCTION = "select"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("reference_audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_slot": ("STRING", {"forceInput": True}),
            },
            "optional": {
                f"speaker_{slot}": ("AUDIO", {"lazy": True}) for slot in VOICE_SLOTS
            },
        }

    @staticmethod
    def _input_name(voice_slot: Any) -> str:
        slot = str(_first(voice_slot, "")).strip().upper()
        if slot not in VOICE_SLOTS:
            raise ValueError(
                f"Invalid or missing voice slot '{slot}'. Expected one of {', '.join(VOICE_SLOTS)}."
            )
        return f"speaker_{slot}"

    def check_lazy_status(self, voice_slot, **speaker_audio):
        input_name = self._input_name(voice_slot)
        return [input_name] if speaker_audio.get(input_name) is None else []

    def select(self, voice_slot, **speaker_audio):
        input_name = self._input_name(voice_slot)
        audio = speaker_audio.get(input_name)
        if audio is None:
            raise ValueError(
                f"Voice slot {input_name[-1]} is used by this chapter, but its reference audio is not connected."
            )
        return (_first(audio),)


class NWFDialogueAudioGate:
    CATEGORY = "Novel Workflow"
    FUNCTION = "fit"
    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "frame_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "target_duration": ("FLOAT", {"forceInput": True}),
                "silence_sample_rate": (
                    "INT",
                    {"default": 24000, "min": 8000, "max": 48000, "step": 1000},
                ),
            },
            "optional": {
                "speech_audio": ("AUDIO", {"lazy": True}),
            },
        }

    def check_lazy_status(
        self,
        text,
        target_duration,
        silence_sample_rate,
        speech_audio=None,
        max_dialogue_seconds=S2V_DEFAULT_MAX_DIALOGUE_SECONDS,
    ):
        del target_duration, silence_sample_rate, max_dialogue_seconds
        needs_speech = bool(str(_first(text, "")).strip())
        return ["speech_audio"] if needs_speech and speech_audio is None else []

    def fit(
        self,
        text,
        target_duration,
        silence_sample_rate,
        speech_audio=None,
        max_dialogue_seconds=S2V_DEFAULT_MAX_DIALOGUE_SECONDS,
    ):
        text = str(_first(text, "")).strip()
        target_duration = float(_first(target_duration, 5.0))
        max_dialogue_seconds = float(
            _first(max_dialogue_seconds, S2V_DEFAULT_MAX_DIALOGUE_SECONDS)
        )
        if not text:
            frame_count = _s2v_frame_count(target_duration)
            duration = frame_count / S2V_FPS
            return (_silence_audio(duration, int(_first(silence_sample_rate, 24000))), frame_count)

        audio = _first(speech_audio)
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("CosyVoice3 did not return a valid ComfyUI AUDIO value.")
        waveform = audio["waveform"]
        if waveform.numel() == 0 or float(waveform.abs().max()) < 1e-7:
            raise RuntimeError(
                "CosyVoice3 returned silent audio for non-empty dialogue. Check the model, reference audio, "
                "and ComfyUI console for the original synthesis error."
            )

        speech_duration = waveform.shape[-1] / int(audio["sample_rate"])
        if speech_duration > max_dialogue_seconds:
            raise ValueError(
                f"Dialogue audio is {speech_duration:.2f}s, longer than the configured "
                f"{max_dialogue_seconds:.2f}s single-shot limit. Split the line, increase "
                "CosyVoice3 speed, or raise max_dialogue_seconds if GPU memory allows."
            )
        required_duration = max(target_duration, speech_duration + 0.25)
        requested_frames = max(S2V_MIN_FRAMES, math.ceil(required_duration * S2V_FPS))
        frame_count = 1 + 4 * math.ceil((requested_frames - 1) / 4)
        duration = frame_count / S2V_FPS
        return (_fit_audio_duration(audio, duration), frame_count)


class NWFConcatVideos:
    CATEGORY = "Novel Workflow"
    FUNCTION = "concat"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("VIDEO",),
                "crossfade_frames": (
                    "INT",
                    {"default": 4, "min": 0, "max": 48, "step": 1},
                ),
            },
            "optional": {
                "transitions": ("STRING", {"forceInput": True}),
            },
        }

    def concat(self, videos, crossfade_frames, transitions=None):
        videos = list(videos or [])
        if not videos:
            raise ValueError("No shot videos were provided for concatenation.")

        components = [video.get_components() for video in videos]
        fps_values = [float(component.frame_rate) for component in components]
        if max(fps_values) - min(fps_values) > 1e-6:
            raise ValueError(f"All clips must use the same frame rate. Got {fps_values}.")
        transitions = [str(item) for item in (transitions or [])]

        frames = _concat_frame_batches(
            [component.images for component in components],
            int(_first(crossfade_frames, 0)),
            transitions,
        )
        audio = _concat_audio_batches(
            [component.audio for component in components],
            [component.images.shape[0] for component in components],
            fps_values[0],
            transitions,
            int(_first(crossfade_frames, 0)),
        )
        bit_depth = videos[0].get_bit_depth() if hasattr(videos[0], "get_bit_depth") else 8

        from comfy_api.latest import InputImpl, Types

        output_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=frames,
                audio=audio,
                frame_rate=Fraction(components[0].frame_rate),
            ),
            bit_depth=bit_depth,
        )
        return (output_video,)


class NWFSaveText:
    CATEGORY = "Novel Workflow"
    FUNCTION = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
                "filename_prefix": ("STRING", {"default": "novel/storyboard"}),
            }
        }

    @staticmethod
    def _safe_target(output_root: Path, filename_prefix: str) -> Path:
        parts = []
        for part in re.split(r"[\\/]+", filename_prefix.strip()):
            if not part or part in {".", ".."}:
                continue
            cleaned = re.sub(r"[<>:\"|?*\x00-\x1f]", "_", part).strip(" .")
            if cleaned:
                parts.append(cleaned)
        if not parts:
            parts = ["storyboard"]

        directory = output_root.joinpath(*parts[:-1])
        directory.mkdir(parents=True, exist_ok=True)
        stem = parts[-1]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        candidate = directory / f"{stem}_{timestamp}.json"
        counter = 1
        while candidate.exists():
            candidate = directory / f"{stem}_{timestamp}_{counter:02d}.json"
            counter += 1
        return candidate

    def save(self, text, filename_prefix):
        output_root = (
            Path(folder_paths.get_output_directory())
            if folder_paths is not None
            else Path.cwd() / "output"
        )
        target = self._safe_target(output_root, filename_prefix)
        target.write_text(text, encoding="utf-8")
        return {"ui": {"text": [str(target)]}, "result": ()}


NODE_CLASS_MAPPINGS = {
    "NWFNovelChapterPlanner": NWFNovelChapterPlanner,
    "NWFTextToSpeech": NWFTextToSpeech,
    "NWFSelectVoiceReference": NWFSelectVoiceReference,
    "NWFDialogueAudioGate": NWFDialogueAudioGate,
    "NWFConcatVideos": NWFConcatVideos,
    "NWFSaveText": NWFSaveText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NWFNovelChapterPlanner": "Novel Chapter to Storyboard",
    "NWFTextToSpeech": "Dialogue Text to Speech",
    "NWFSelectVoiceReference": "Select Character Voice Reference",
    "NWFDialogueAudioGate": "Dialogue Audio Gate and Duration",
    "NWFConcatVideos": "Concatenate Shot Videos",
    "NWFSaveText": "Save Storyboard JSON",
}
