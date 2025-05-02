from flask import Flask
from router import router

app = Flask(__name__)
app.register_blueprint(router)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
