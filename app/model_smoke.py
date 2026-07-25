from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import replace
from typing import Any, Dict, Sequence

from .config import Settings
from .credentials import CredentialCipher
from .db import Database
from .deepseek import DeepSeekAnalyzer
from .model_provider import settings_for_credential
from .writing import DeepSeekWriter


SAFE_SMOKE_SYSTEM_PROMPT = (
    "这是 novelAI 的合成连通性测试。只处理本次请求提供的虚构材料，"
    "不要引用或假定任何真实作品内容。"
)


def synthetic_writing_context() -> Dict[str, Any]:
    return {
        "chapter": {
            "project_title": "纸灯塔",
            "genre": "温和悬疑",
            "premise": "一名修表师从失效船票中发现旧友留下的约定。",
            "point_of_view": "第三人称限知",
            "world_setting": "当代海边小城，无超自然设定。",
            "style_guide": "克制、具体，以动作呈现情绪。",
            "position": 1,
            "title": "停走的秒针",
            "outline": "修表师核对船票日期，决定去旧码头查证。",
            "key_points": "发现日期与记忆不符\n带上船票出门",
            "target_chapter_chars": 260,
            "ai_instructions": "",
        },
        "characters": [
            {
                "name": "林岚",
                "role": "修表师",
                "traits": "谨慎、敏锐",
            }
        ],
        "canonical_memory": {
            "source": "author_confirmed_canon_only"
        },
        "task_card": {
            "purpose": "让主角主动开始调查",
            "must_happen": ["核对日期", "决定前往旧码头"],
            "target_chars": 260,
            "scenes": [],
        },
    }


def load_personal_model_settings(
    settings: Settings, username: str
) -> Settings:
    database = Database(settings.database_path)
    user = database.get_user_by_username(username)
    if not user:
        raise ValueError(f"账号不存在：{username}")
    credential = database.get_api_credential(int(user["id"]))
    if not credential:
        raise ValueError(f"账号尚未配置个人模型：{username}")
    api_key = CredentialCipher(settings.credential_secret).decrypt(
        str(credential["encrypted_key"])
    )
    personal = settings_for_credential(
        settings,
        credential=credential,
        api_key=api_key,
    )
    return replace(
        personal,
        deepseek_system_prompt=SAFE_SMOKE_SYSTEM_PROMPT,
        deepseek_max_tokens=min(personal.deepseek_max_tokens, 3_000),
        deepseek_max_retries=min(personal.deepseek_max_retries, 1),
    )


async def run_smoke(
    settings: Settings, *, run_text: bool = True, run_json: bool = True
) -> Sequence[Dict[str, Any]]:
    results = []
    provider_user_id = "novelai-synthetic-smoke"
    if run_text:
        writer = DeepSeekWriter(settings)
        try:
            started = time.perf_counter()
            response = await writer.write(
                context=synthetic_writing_context(),
                operation="draft",
                instruction="这是合成冒烟测试，控制在一个短场景内。",
                current_content="",
                previous_content="",
                provider_user_id=provider_user_id,
            )
            results.append(
                {
                    "task": "text",
                    "status": "pass",
                    "provider": writer.provider,
                    "model": writer.model,
                    "chars": len(response.content),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
        finally:
            await writer.close()

    if run_json:
        analyzer = DeepSeekAnalyzer(settings)
        try:
            started = time.perf_counter()
            response = await analyzer.analyze(
                "停走的秒针",
                "林岚把旧船票压在工作灯下。票面日期比她记忆中的约定"
                "早了整整一天。她翻出修理簿核对，发现旧友送表的记录"
                "也写着同一个日期。傍晚，她合上店门，把船票装进"
                "内袋，朝废弃的三号码头走去。",
                provider_user_id,
            )
            results.append(
                {
                    "task": "json",
                    "status": "pass",
                    "provider": analyzer.provider,
                    "model": analyzer.model,
                    "summary_chars": len(response.result.summary),
                    "events": len(response.result.key_events),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
        finally:
            await analyzer.close()
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "用内置合成内容验证账号的模型文本与 JSON 能力；"
            "不读取作品正文，也不打印 API Key 或模型输出。"
        )
    )
    parser.add_argument("--username", required=True, help="本地账号名")
    parser.add_argument(
        "--mode",
        choices=("all", "text", "json"),
        default="all",
        help="要验证的能力，默认 all",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = load_personal_model_settings(
        Settings.from_env(), args.username
    )
    results = asyncio.run(
        run_smoke(
            settings,
            run_text=args.mode in {"all", "text"},
            run_json=args.mode in {"all", "json"},
        )
    )
    for result in results:
        metrics = " ".join(
            f"{key}={value}" for key, value in result.items()
        )
        print(metrics)


if __name__ == "__main__":
    main()
