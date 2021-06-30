import os
import sys
import logging
import random
from datetime import datetime
from flask import Flask, request, render_template, jsonify, abort, Response, send_from_directory
# from flask_cors import CORS
# import psycopg2

import sandbox

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
# CORS(app)

@app.before_first_request
def initialize():
    sandbox.setup()

# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory('client/public', 'index.html')

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('client/public', path)


@app.route("/getGender/<featureName>")
def getGender(featureName):
    featureName = request.view_args['featureName']
    return (sandbox.test_plot(featureName))

@app.route("/getBefore")
def getBefore():
    return (sandbox.plotBefore())

@app.route("/injectBias")
def injectBias():
    return (sandbox.injectBias())

@app.route("/injectBiasUnder")
def injectBiasUnder():
    return (sandbox.injectBiasUnder())


if __name__ == "__main__":
    app.run(debug=True)
