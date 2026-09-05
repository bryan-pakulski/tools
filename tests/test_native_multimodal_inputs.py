from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from mu.artifact import ArtifactRegistry
from mu.session.media import media_resolver_for_session, tool_media_references
from mu.session.messages import build_messages_from_history
from providers.base import LLMProvider, MediaData, Message, MessagePart
from providers.gemini import GeminiProvider
from providers.ollama import OllamaProvider
from providers.openai import OpenAIProvider, _media_content_part


PNG = b"\x89PNG\r\n\x1a\nnative-pixels"


def _openai(model: str = "gpt-5.6-terra") -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    LLMProvider.__init__(provider, model)
    provider.name = "openai"
    return provider


def _gemini(model: str = "gemini-3.6-flash") -> GeminiProvider:
    provider = GeminiProvider.__new__(GeminiProvider)
    LLMProvider.__init__(provider, model)
    provider.name = "gemini"
    return provider


def test_openai_chat_completions_native_content_shapes():
    image = _media_content_part(
        MediaData(PNG, "image/png", display_name="screen.png")
    )
    assert image == {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
        },
    }

    audio = _media_content_part(
        MediaData(b"wave", "audio/wav", display_name="note.wav")
    )
    assert audio == {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(b"wave").decode("ascii"),
            "format": "wav",
        },
    }

    document = _media_content_part(
        MediaData(b"%PDF", "application/pdf", display_name="report.pdf")
    )
    assert document["type"] == "file"
    assert document["file"]["filename"] == "report.pdf"
    assert document["file"]["file_data"].startswith(
        "data:application/pdf;base64,"
    )


def test_openai_tool_media_follows_tool_messages_as_native_user_content():
    converted = _openai()._convert_messages(
        [
            Message(
                role="tool",
                parts=[
                    MessagePart(
                        type="tool_result",
                        tool_name="browser_snapshot",
                        tool_call_id="call-1",
                        tool_result={"ok": True},
                        media_inputs=[
                            MediaData(PNG, "image/png", display_name="snapshot.png")
                        ],
                    )
                ],
            )
        ]
    )
    assert converted[0]["role"] == "tool"
    assert converted[0]["tool_call_id"] == "call-1"
    assert converted[1]["role"] == "user"
    assert converted[1]["content"][1]["type"] == "image_url"


def test_gemini_uses_inline_blob_for_uploaded_audio_and_video():
    converted = _gemini()._convert_to_gemini_contents(
        [
            Message(
                role="user",
                parts=[
                    MessagePart(
                        type="media_input",
                        media=MediaData(
                            b"audio", "audio/mpeg", display_name="clip.mp3"
                        ),
                    ),
                    MessagePart(
                        type="media_input",
                        media=MediaData(
                            b"video", "video/mp4", display_name="clip.mp4"
                        ),
                    ),
                ],
            )
        ]
    )
    parts = converted[0].parts
    assert [part.inline_data.mime_type for part in parts] == [
        "audio/mpeg",
        "video/mp4",
    ]
    assert [part.inline_data.data for part in parts] == [b"audio", b"video"]


def test_gemini_3_nests_tool_media_in_function_response_parts():
    converted = _gemini()._convert_to_gemini_contents(
        [
            Message(
                role="tool",
                parts=[
                    MessagePart(
                        type="tool_result",
                        tool_name="browser_snapshot",
                        tool_result="captured",
                        media_inputs=[
                            MediaData(PNG, "image/png", display_name="screen.png")
                        ],
                    )
                ],
            )
        ]
    )
    response = converted[0].parts[0].function_response
    assert response.response["media"][0]["$ref"].startswith("tool-media-0-")
    assert response.parts[0].inline_data.mime_type == "image/png"
    assert response.parts[0].inline_data.data == PNG


def test_ollama_glm_53_uses_rest_images_array_for_native_pixels():
    provider = OllamaProvider("glm-5.3-flash:cloud", host="https://ollama.com")
    converted = provider._convert_messages(
        [
            Message(
                role="user",
                parts=[
                    MessagePart(type="text", text="inspect"),
                    MessagePart(
                        type="media_input",
                        media=MediaData(PNG, "image/png", display_name="screen.png"),
                    ),
                ],
            )
        ]
    )
    assert converted[0]["images"] == [base64.b64encode(PNG).decode("ascii")]


def test_ollama_unknown_local_model_uses_api_show_vision_capability(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.read.return_value = json.dumps(
        {"capabilities": ["completion", "vision"]}
    ).encode("utf-8")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)

    provider = OllamaProvider("local-vision", host="http://ollama.test")
    assert provider.supports_input_mime("image/png", "screen.png") is True
    assert provider.supports_input_mime("audio/wav", "audio.wav") is False


def test_tool_artifact_reference_resolves_original_pixels_without_ocr(tmp_path):
    registry = ArtifactRegistry(str(tmp_path / "session"))
    descriptor = registry.add(
        "snapshot.png", content=PNG, mime_type="image/png", kind="file"
    )
    provider = OllamaProvider("glm-5.3-flash:cloud", host="https://ollama.com")
    session = SimpleNamespace(provider=provider, artifact_registry=registry)

    references = tool_media_references(
        session, {"ok": True, "artifacts": [descriptor]}
    )
    assert references == [
        {
            "artifact_id": descriptor["artifact_id"],
            "name": "snapshot.png",
            "mime_type": "image/png",
            "size": len(PNG),
        }
    ]
    media = media_resolver_for_session(session)(references[0])
    assert media is not None
    assert media.data == PNG

    # Composed Chromium tools may return only data.artifact_id; that still
    # resolves to the same native payload.
    fallback = tool_media_references(
        session, {"ok": True, "data": {"artifact_id": descriptor["artifact_id"]}}
    )
    assert fallback == references

    messages = build_messages_from_history(
        [
            {
                "role": "tool",
                "parts": [
                    {
                        "type": "tool_result",
                        "tool_name": "browser_snapshot",
                        "tool_result": {"ok": True},
                        "media_inputs": references,
                    }
                ],
            }
        ],
        {"role": "system", "parts": []},
        tool_result_floor=1,
        media_resolver=media_resolver_for_session(session),
    )
    assert messages[0].parts[0].media_inputs[0].data == PNG


def test_text_only_model_does_not_promote_tool_artifact(tmp_path):
    class TextOnly(LLMProvider):
        def get_available_models(self):
            return []

        def generate(self, *args, **kwargs):
            raise NotImplementedError

        def upload_file(self, file_path, mime_type):
            return None

    provider = TextOnly("unknown")
    provider.name = "custom"
    registry = ArtifactRegistry(str(tmp_path / "session"))
    descriptor = registry.add(
        "snapshot.png", content=PNG, mime_type="image/png", kind="file"
    )
    session = SimpleNamespace(provider=provider, artifact_registry=registry)
    assert tool_media_references(
        session, {"ok": True, "artifacts": [descriptor]}
    ) == []
