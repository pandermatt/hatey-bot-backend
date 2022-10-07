from werkzeug.exceptions import Unauthorized, BadRequest

from util.logger import log


def unauthorized_response():
    log.error(f'Unauthorized')
    raise Unauthorized()


def bad_request_response(message=None):
    log.error(f'BadRequest')
    if message is not None:
        log.error(f'with: {message}')
        raise BadRequest(message)
    raise BadRequest()
