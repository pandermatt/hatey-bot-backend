import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource

from api import error_handler
from api.auth import token_auth
from config import config
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
        return jsonify(
            status='success',
            labels=["hate_speech", "racist", "anti-caucasian", "intelligence discrimating"]
        )


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
