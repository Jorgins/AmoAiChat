"""
Небольшой CLI для ручной проверки сервиса без Postman/curl.
Имитирует диалоговое окно AmoCRM: копит историю переписки в памяти
и на каждое сообщение клиента печатает ответ + подсказку менеджеру.

Запуск: сначала `python app.py` (или uvicorn) в одном терминале,
затем `python test_client.py` в другом.
"""

import requests

API_URL = "http://localhost:8000/process-message"


def main():
    history = []
    print("Имитация переписки с клиентом в AmoCRM. Ctrl+C для выхода.\n")
    while True:
        try:
            message = input("Клиент: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not message:
            continue

        payload = {"client_message": message, "dialog_history": history}
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        print(f"\n>> Ответ клиенту:\n{data['reply_to_client']}")
        if data["upsell_hint"]:
            print(f"\n>> Подсказка менеджеру (допродажа):\n{data['upsell_hint']}")
        print()

        history.append({"role": "client", "text": message})
        history.append({"role": "manager", "text": data["reply_to_client"]})


if __name__ == "__main__":
    main()
