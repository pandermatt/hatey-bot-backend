import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource
from slack import WebClient
from slackeventsapi import SlackEventAdapter

from api import error_handler
from api.auth import token_auth
from config import config
from core.hatey_predictor import hatey_predictor_singleton
from util.logger import log

app = Flask(__name__)
app_origins = {"origins": config.get_env("FRONTEND_URL")}
CORS(
    app,
    resources={
        "/auth/": app_origins,
        "/queries/*": app_origins,
    }
)
authorizations = {
    'Bearer Auth': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization'
    },
}

api = Api(app, version='0.0.1', title='HateyBot API',
          description='API for HateyBot',
          security='Bearer Auth',
          authorizations=authorizations,
          doc='/'
          )

SLACK_INTEGRATION_ENABLED = config.get_env("SLACK_BOT_TOKEN") and config.get_env("SLACK_SIGNING_SECRET")
if SLACK_INTEGRATION_ENABLED:
    slack_client = WebClient(token=config.get_env('SLACK_BOT_TOKEN'))
    slack_adapter = SlackEventAdapter(config.get_env('SLACK_SIGNING_SECRET'), '/slack/events', app)
    slack_bot_id = slack_client.api_call("auth.test")["user_id"]
else:
    log.warning("SLACK_BOT_TOKEN or SLACK_SIGNING_SECRET not set, Slack integration disabled")

auth = api.namespace('auth', description='Authentication')
queries = api.namespace('queries', description='Queries')


@auth.route('/')
class AuthHandler(Resource):
    @token_auth.login_required
    def get(self):
        return 'successful'


@queries.route('/')
@queries.doc(params={'query': 'Query to classify'})
class QueryList(Resource):
    @token_auth.login_required
    def get(self):
        params = request.args
        query = params.get('query')
        if query is None:
            error_handler.bad_request_response('Query is required')
        log.info(f'QueryList: {params}')

        try:
            return jsonify(
                predictions=hatey_predictor_singleton.predictions(query),
                sentiment=hatey_predictor_singleton.sentiment(query),
                is_hate_speech=hatey_predictor_singleton.is_hate_speech(query),
                problematic_words=hatey_predictor_singleton.problematic_words(query),
                reasons=hatey_predictor_singleton.reasons(query)
            )
        except Exception as e:
            log.error(e)
            return jsonify(status='error')


if SLACK_INTEGRATION_ENABLED:
    @slack_adapter.on('message')
    def message(payload):
        event = payload.get('event', {})
        channel_id = event.get('channel')
        user_id = event.get('user')
        text = event.get('text')

        if slack_bot_id == user_id:
            return

        if hatey_predictor_singleton.is_hate_speech(text):
            text = f"Hey <@{user_id}>! I think your message is hate speech " \
                   f"[{hatey_predictor_singleton.reasons_as_text(text)}]. " \
                   f"Please use more appropriate language."
            log.info(f"Sending message to channel {channel_id}: {text}")

            slack_client.chat_postMessage(channel=channel_id, text=text)


@app.after_request
def after_request(response):
    log.info('%s %s %s %s %s', request.remote_addr, request.method, request.scheme, request.full_path, response.status)
    return response


@app.errorhandler(Exception)
def exceptions(e):
    tb = traceback.format_exc()
    log.error('%s %s %s %s 5xx INTERNAL SERVER ERROR\n%s', request.remote_addr, request.method, request.scheme,
              request.full_path, tb)
    return e.status_code


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404


if __name__ == '__main__':
    app.run(threaded=False, debug=False)
