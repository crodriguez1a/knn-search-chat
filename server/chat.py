import os
from cmd import Cmd
import requests

from app.chat import bubbles

default_flask: str = 'http://127.0.0.1:5000'
CHAT_SERVER: str = os.getenv('CHAT_SERVER_URL', default_flask)


class ChatPrompt(Cmd):
    def do_exit(self, message) -> bool:
        print("Bye")
        return True

    def do_say(self, message: str) -> dict:
        bubbles(5)
        response = requests.post(
            f'{CHAT_SERVER}/chat',
            data={
                'message': message})
        print(f"Bot says: {response.text}")


ChatPrompt().cmdloop()
