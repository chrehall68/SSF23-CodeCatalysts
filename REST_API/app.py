from flask import Flask, request, redirect
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import prediction

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


class Test(Resource):
    def get(self):
        return "Welcome to, Hate Speech Detection API!"

    def post(self):
        try:
            value = request.get_json()
            if value:
                return {"Post Values": value}, 201

            return {"error": "Invalid format."}

        except Exception as error:
            return {"error": error}


class GetPredictionOutput(Resource):
    def get(self):
        return {"error": "Invalid Method."}

    def post(self):
        try:
            data = request.get_json()
            predict = prediction.predict(data["text"])
            predictOutput = predict
            return {"predict": predictOutput}

        except Exception as error:
            return {"error": str(error)}


api.add_resource(Test, "/")
api.add_resource(GetPredictionOutput, "/predict")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
