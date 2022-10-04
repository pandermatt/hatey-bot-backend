import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource

from util.logger import log

app = Flask(__name__)
CORS(
    app,
    resources={
        "/auth/": {"origins": ["http://localhost:8080"]},
    }
)
authorizations = {
    'Bearer Auth': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization'
    },
}

swagger_ui_enabled = '/'

api = Api(app, version='0.0.1', title='HateyBot API',
          description='API for HateyBot',
          security='Bearer Auth',
          authorizations=authorizations,
          doc=swagger_ui_enabled
          )

auth = api.namespace('auth', description='Authentication')


@auth.route('/')
class AuthHandler(Resource):
    def get(self):
        return 'successful'


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
