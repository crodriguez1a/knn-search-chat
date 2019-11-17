import logging

from flask import Flask, request
app = Flask(__name__)

from server.middleware import BotMiddleWare, encode, bot, search_index

# initialize middleware
bot_middleware: BotMiddleWare = BotMiddleWare(encode, bot, search_index)


@app.route('/chat', methods=['POST'])
def chat():

    if request.method == 'POST':
        try:
            message: str = request.form['message']
            logging.info(message)
            return bot_middleware.post(message)
        except Exception as e:
            # TODO proper error handling
            raise e

    return ''


if __name__ == '__main__':
    app.run()
