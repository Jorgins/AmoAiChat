"""
AmoAiChat — простой сервис-помощник для менеджера в AmoCRM.

Принимает сообщение клиента (и, опционально, историю переписки из диалогового
окна AmoCRM), сверяется с базой знаний (knowledge_base.json) и возвращает:
  1. reply_to_client   — готовый вежливый ответ клиенту
  2. upsell_hint       — подсказку менеджеру по возможному допродажи

Интеграция с AmoCRM (вебхуки/API) сюда не входит — сервис принимает текст
как обычный JSON и может быть подключён к вебхуку AmoCRM отдельным шагом.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

MODEL_ID = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
KB_PATH = Path(__file__).parent / "knowledge_base.json"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AmoAiChat", description="Ответ клиенту + подсказка по допродаже для менеджера AmoCRM")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
client = anthropic.Anthropic()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def load_knowledge_base() -> dict:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


KNOWLEDGE_BASE = load_knowledge_base()


class DialogTurn(BaseModel):
    role: str = Field(..., description="'client' или 'manager'")
    text: str


class ProcessRequest(BaseModel):
    client_message: str = Field(..., description="Текущее сообщение клиента")
    dialog_history: Optional[List[DialogTurn]] = Field(
        default=None,
        description="История переписки из диалогового окна AmoCRM (по возрастанию времени)",
    )
    client_name: Optional[str] = Field(default=None, description="Имя клиента, если известно")


class ProcessResponse(BaseModel):
    reply_to_client: str
    upsell_hint: str
    upsell_suggestions: List[str] = Field(default_factory=list)


RESPONSE_TOOL = {
    "name": "give_response",
    "description": "Вернуть готовый ответ клиенту и подсказку по допродаже для менеджера",
    "input_schema": {
        "type": "object",
        "properties": {
            "reply_to_client": {
                "type": "string",
                "description": "Вежливый, короткий ответ клиенту на его сообщение, основанный ТОЛЬКО на базе знаний",
            },
            "upsell_hint": {
                "type": "string",
                "description": "Короткая подсказка-объяснение МЕНЕДЖЕРУ: что происходит и почему стоит (или пока не стоит) предлагать допродажу (или пустая строка)",
            },
            "upsell_suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "0-3 коротких готовых фразы для допродажи в форме, пригодной сразу отправить "
                    "клиенту (менеджер вставляет их одним кликом в своё сообщение). Каждая — про ОДИН "
                    "конкретный сопутствующий товар/услугу. Пустой массив, если предлагать пока нечего."
                ),
            },
        },
        "required": ["reply_to_client", "upsell_hint", "upsell_suggestions"],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_system_prompt(kb: dict, is_first_message: bool) -> str:
    intro_rule = (
        f"Это ПЕРВОЕ сообщение клиента в диалоге (редкий случай — обычно приветствие уже отправлено "
        f"заранее). Если приветствия ещё не было, поздоровайся и коротко представься в естественной форме, "
        f"например: «Здравствуйте! Я консультант {kb['assistant_name']} — отвечу на все ваши вопросы 🙂», "
        f"и только потом переходи к сути."
        if is_first_message
        else
        f"Ты уже поздоровался и представился ранее (это сделано автоматически до этого сообщения) — "
        f"повторно представляться не нужно, сразу отвечай по существу."
    )
    return f"""Ты — {kb['assistant_name']}, консультант отдела продаж компании «{kb['company_name']}», встроенный в переписку с клиентом в AmoCRM. Для клиента ты — единственный человек, с которым он общается, по имени {kb['assistant_name']}; никогда не упоминай слово "менеджер" в разговоре с клиентом и не перекладывай вопросы на менеджера — ты сам всё решаешь и уточняешь (наличие на складе, сроки и т.д.), от первого лица ("я уточню", "могу проверить"), а не "менеджер вам ответит" / "спрошу у менеджера".

{intro_rule}

Твоя задача на каждое обращение клиента — подготовить два результата и вызвать инструмент give_response:

1. reply_to_client — вежливый ответ клиенту.
   - Используй ТОЛЬКО факты из базы знаний ниже. Не придумывай сроки, цены и условия, которых там нет.
   - Если в базе знаний нет ответа на вопрос — вежливо сообщи, что уточнишь детали, и не выдумывай факты.
   - НИКОГДА не пиши "менеджер уточнит", "вам ответит менеджер", "передам менеджеру" — отвечай от
     своего лица, ты сам ведёшь весь диалог.
   - КРИТИЧЕСКИ ВАЖНО: задавай НЕ БОЛЕЕ ОДНОГО вопроса за сообщение. Никогда не перечисляй несколько
     вопросов списком — только один, и жди ответа клиента, прежде чем задать следующий.
   - Правило уточняющих вопросов: {kb['clarification_policy']}
   - Тон: {kb['tone_guidelines']}

2. upsell_hint — короткая подсказка-ОБЪЯСНЕНИЕ менеджеру (не клиенту!): что происходит и почему стоит
   (или пока не стоит) предлагать допродажу, опираясь на историю диалога и upsell_playbook. Если клиент
   ещё не определился с параметрами товара и ты задаёшь уточняющий вопрос — объясни менеджеру, что нужно
   продолжить диалог, и какие сопутствующие товары держать в голове на будущее. Если уместного апсейла
   пока нет вообще — оставь пустую строку.

3. upsell_suggestions — 0-3 КОРОТКИЕ готовые фразы (не более одного предложения каждая), которые менеджер
   сможет вставить одним кликом прямо в сообщение клиенту. Формулируй их от лица {kb['assistant_name']}
   так, будто это реальное продолжение диалога с клиентом.
   - ТОЛЬКО в форме ненавязчивого ВОПРОСА, никогда не как прямая продажа/предложение "в лоб". Спрашивай,
     а не предлагай: «А у вас уже есть силовой кабель для подключения?», «Вам, кстати, для этого генератора
     провода не нужны?», «А моторное масло на первое ТО у вас есть, или подсказать какое взять?».
   - НЕ пиши "рекомендую взять", "предлагаем купить", "советую приобрести" и т.п. — это должно звучать как
     обычная забота продавца-консультанта, а не навязывание.
   - Предлагай их ТОЛЬКО когда это уместно по upsell_playbook (клиент уже определился с товаром/моделью) —
     иначе оставь пустой массив. Никогда не более 1-2 фраз за раз (см. правило upsell_playbook про
     количество апсейлов).

База знаний (тематики и факты):
{json.dumps(kb['topics'], ensure_ascii=False, indent=2)}

Общие правила допродаж (upsell_playbook):
{json.dumps(kb['upsell_playbook'], ensure_ascii=False, indent=2)}

Всегда вызывай инструмент give_response ровно один раз."""


def format_dialog_history(history: Optional[List[DialogTurn]]) -> str:
    if not history:
        return "(истории переписки нет — это первое сообщение)"
    lines = []
    for turn in history:
        speaker = "Клиент" if turn.role == "client" else "Менеджер"
        lines.append(f"{speaker}: {turn.text}")
    return "\n".join(lines)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/process-message", response_model=ProcessResponse)
def process_message(req: ProcessRequest):
    is_first_message = not req.dialog_history
    user_content = (
        f"История переписки в AmoCRM:\n{format_dialog_history(req.dialog_history)}\n\n"
        f"Новое сообщение клиента"
        f"{f' ({req.client_name})' if req.client_name else ''}:\n{req.client_message}"
    )

    request_kwargs = dict(
        model=MODEL_ID,
        max_tokens=1024,
        system=build_system_prompt(KNOWLEDGE_BASE, is_first_message),
        tools=[RESPONSE_TOOL],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": user_content}],
    )
    if MODEL_ID.startswith("claude-"):
        # effort-параметр понимает только нативный Claude API, не прокси-совместимые бэкенды
        request_kwargs["output_config"] = {"effort": "low"}

    try:
        response = client.messages.create(**request_kwargs)
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка обращения к LLM API: {e.message}")

    for block in response.content:
        if block.type == "tool_use" and block.name == "give_response":
            data = block.input
            return ProcessResponse(
                reply_to_client=data["reply_to_client"],
                upsell_hint=data.get("upsell_hint", ""),
                upsell_suggestions=data.get("upsell_suggestions", []),
            )

    raise HTTPException(status_code=502, detail="Модель не вернула ожидаемый ответ (tool_use отсутствует)")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
