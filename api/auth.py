from flask_httpauth import HTTPTokenAuth

from api.error_handler import unauthorized_response
from config import config

token_auth = HTTPTokenAuth()


@token_auth.verify_token
def verify_token(token):
    try:
        return token == config.get_env('API_TOKEN')
    except KeyError:
        token_auth_error()


@token_auth.error_handler
def token_auth_error():
    return unauthorized_response()
