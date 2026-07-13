from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

try:
    import folder_paths
except ImportError:  # Allows lightweight unit tests outside a ComfyUI install.
    folder_paths = None


MAX_SEED = 2**53 - 1


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

    shots = []
    for index, raw in enumerate(raw_shots[:max_shots]):
        if not isinstance(raw, dict):
            continue

        image_prompt = str(raw.get("image_prompt", "")).strip()
        video_prompt = str(raw.get("video_prompt", "")).strip()
        if not image_prompt or not video_prompt:
            continue

        if style_prompt and style_prompt.lower() not in image_prompt.lower():
            image_prompt = f"{image_prompt}. Global visual style: {style_prompt}"
        if avoid_prompt:
            image_prompt = f"{image_prompt}. Exclude: {avoid_prompt}"
        if "identity" not in video_prompt.lower():
            video_prompt = (
                f"{video_prompt}. Maintain exact character identity, facial features, costume, "
                "scene layout, and visual style from the start image."
            )

        try:
            duration = float(raw.get("duration_seconds", seconds_per_shot))
        except (TypeError, ValueError):
            duration = seconds_per_shot
        duration = round(max(2.0, min(10.0, duration)), 2)

        shot_id = str(raw.get("id") or f"S{index + 1:03d}").strip()
        shots.append(
            {
                "id": shot_id,
                "title": str(raw.get("title", f"Shot {index + 1}")).strip(),
                "duration_seconds": duration,
                "image_prompt": image_prompt,
                "video_prompt": video_prompt,
                "narration_or_dialogue": str(raw.get("narration_or_dialogue", "")).strip(),
                "seed": int((base_seed + index * 104729) % MAX_SEED),
            }
        )

    if not shots:
        raise ValueError("No valid shots with both image_prompt and video_prompt were returned.")

    return {
        "project": payload.get("project", {}),
        "continuity_bible": payload.get("continuity_bible", {}),
        "shots": shots,
    }


def _concat_frame_batches(frame_batches: list[torch.Tensor], crossfade_frames: int) -> torch.Tensor:
    if not frame_batches:
        raise ValueError("At least one frame batch is required.")

    result = frame_batches[0]
    if result.ndim != 4:
        raise ValueError("Video frames must use ComfyUI's [frames, height, width, channels] shape.")

    for next_frames in frame_batches[1:]:
        if next_frames.ndim != 4:
            raise ValueError("Video frames must use ComfyUI's [frames, height, width, channels] shape.")
        if result.shape[1:] != next_frames.shape[1:]:
            raise ValueError(
                f"All clips must have the same frame size. Got {tuple(result.shape[1:])} "
                f"and {tuple(next_frames.shape[1:])}."
            )

        overlap = min(max(0, crossfade_frames), result.shape[0], next_frames.shape[0])
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


class NWFNovelChapterPlanner:
    CATEGORY = "Novel Workflow"
    FUNCTION = "plan"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "image_prompts",
        "video_prompts",
        "shot_ids",
        "seeds",
        "durations",
        "storyboard_json",
    )
    OUTPUT_IS_LIST = (True, True, True, True, True, False)

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
                    {"default": 5.0, "min": 2.0, "max": 10.0, "step": 0.5},
                ),
                "style_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "cinematic Chinese live-action drama, realistic skin and fabric, "
                            "production design, natural lighting, 16:9 film still, consistent color grade"
                        ),
                    },
                ),
                "avoid_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "text, subtitles, watermark, logo, duplicate people, malformed hands, "
                            "extra fingers, inconsistent costume, low detail, oversaturated colors"
                        ),
                    },
                ),
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
    def _request_storyboard(
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
        endpoint = api_base.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"

        system_prompt = f"""
You are a film director and storyboard supervisor. Convert one complete Chinese novel chapter into a
production-ready storyboard for an automated image-to-video pipeline.

Return exactly one JSON object and no markdown. The schema is:
{{
  "project": {{"title": "...", "summary": "..."}},
  "continuity_bible": {{
    "visual_style": "...",
    "characters": [{{"name": "...", "immutable_visual_description": "..."}}],
    "locations": [{{"name": "...", "immutable_visual_description": "..."}}]
  }},
  "shots": [
    {{
      "id": "S001",
      "title": "...",
      "duration_seconds": {seconds_per_shot},
      "image_prompt": "English prompt for Z-Image-Turbo",
      "video_prompt": "English motion prompt for Wan 2.2 image-to-video",
      "narration_or_dialogue": "Chinese text or empty string"
    }}
  ]
}}

Rules:
- Use at most {max_shots} shots and cover the chapter from beginning to end in chronological order.
- Prefer visually meaningful beats. Combine adjacent prose that would produce nearly identical shots.
- Establish a strict continuity bible before writing shots.
- Every image_prompt must be self-contained English. Repeat the exact immutable description for every
  visible recurring character; never rely on pronouns or earlier shots.
- Every image_prompt must specify subject, action, environment, shot size, camera angle, composition,
  lighting, mood, and this global style: {style_prompt}
- Keep one dominant visual moment per image. Do not ask the image model to render captions or dialogue.
- Every video_prompt must describe only plausible subject motion, environmental motion, camera motion,
  pace, and what must stay unchanged from the start image. Do not introduce a new character or location.
- Keep each duration between 2 and 10 seconds, close to {seconds_per_shot} seconds.
- Preserve the novel's facts. Do not invent major plot events.
""".strip()

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chapter},
            ],
            "temperature": temperature,
            "stream": False,
            "max_tokens": 12000,
        }
        headers = {"Content-Type": "application/json"}
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
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Storyboard API returned HTTP {error.code}: {details[:1000]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not reach the storyboard API at {endpoint}: {error.reason}"
            ) from error

        try:
            content = response_body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            return _extract_json(str(content))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "The storyboard API response was not valid Chat Completions JSON with a parseable "
                f"storyboard: {str(response_body)[:1000]}"
            ) from error

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
            json.dumps(storyboard, ensure_ascii=False, indent=2),
        )


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
            }
        }

    def concat(self, videos, crossfade_frames):
        videos = list(videos or [])
        if not videos:
            raise ValueError("No shot videos were provided for concatenation.")

        components = [video.get_components() for video in videos]
        fps_values = [float(component.frame_rate) for component in components]
        if max(fps_values) - min(fps_values) > 1e-6:
            raise ValueError(f"All clips must use the same frame rate. Got {fps_values}.")
        if any(component.audio is not None for component in components):
            raise ValueError("Audio concatenation is not supported. Connect silent Wan clips here.")

        frames = _concat_frame_batches(
            [component.images for component in components],
            int(_first(crossfade_frames, 0)),
        )
        bit_depth = videos[0].get_bit_depth() if hasattr(videos[0], "get_bit_depth") else 8

        from comfy_api.latest import InputImpl, Types

        output_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=frames,
                audio=None,
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
    "NWFConcatVideos": NWFConcatVideos,
    "NWFSaveText": NWFSaveText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NWFNovelChapterPlanner": "Novel Chapter to Storyboard",
    "NWFConcatVideos": "Concatenate Shot Videos",
    "NWFSaveText": "Save Storyboard JSON",
}
