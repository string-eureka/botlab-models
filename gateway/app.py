from flask import Flask
import os

def create_app():
    app = Flask(__name__)

    from routes.clip_routes import clip_bp
    from routes.sam_routes import sam_bp

    app.register_blueprint(clip_bp, url_prefix="/clip")
    app.register_blueprint(sam_bp, url_prefix="/sam")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000)
