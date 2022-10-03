from werkzeug.exceptions import BadRequest, Unauthorized, NotFound

from util.logger import log


def unauthorized_response():
    log.error(f'Unauthorized')
    raise Unauthorized()
