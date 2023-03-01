from flask import abort, Flask, request, send_file
from flask import render_template
from werkzeug import utils
from python_coreml_stable_diffusion.pipeline import get_coreml_pipe
from diffusers import StableDiffusionPipeline
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

app = Flask(__name__)

mlpackages_dir = Path(__file__).joinpath('..', 'ml-stable-diffusion', 'output-mlpackages-directory').resolve().absolute()
avatars_dir = Path(__file__).joinpath('..', 'avatars').resolve().absolute()
avatars_dir.mkdir(exist_ok = True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/raw')
def raw():
    username = request.args.get('username')
    if username is None:
        abort(400, 'username is required')
    avatar_path = get_avatar_path(username)
    return send_file(avatar_path)

@app.route('/download', methods=['POST'])
def download():
    username = request.args.get('username')
    if username is None:
        abort(400, 'username is required')
    avatar_path = get_avatar_path(username)
    return send_file(avatar_path, download_name='avatar.png', as_attachment=True)

@app.route('/view')
def view():
    username = request.args.get('username')
    return render_template('avatar.html', username=username)

def get_avatar_path(username):
    filename = utils.secure_filename(username)
    avatar_path = avatars_dir.joinpath("%s.png" % filename).resolve().absolute()
    if avatar_path.exists() != True:
        generate_avatar(username, avatar_path)
    return avatar_path

def generate_avatar(username, avatar_path):
    model_version = "CompVis/stable-diffusion-v1-4"
    np.random.seed(42)
    pytorch_pipe = StableDiffusionPipeline.from_pretrained(model_version,
                                                           use_auth_token=True)

    coreml_pipe = get_coreml_pipe(pytorch_pipe=pytorch_pipe,
                                  mlpackages_dir=mlpackages_dir,
                                  model_version=model_version,
                                  compute_unit="ALL")

    image = coreml_pipe(
        prompt=username,
        height=coreml_pipe.height,
        width=coreml_pipe.width,
        num_inference_steps=50,
        guidance_scale = 8
    )


    if image.nsfw_content_detected:
        img = Image.new('rgb', (512, 512))
        ImageDraw.Draw(img).rectangle([(0,0),img.size], fill=(200,100,200))
        img.save(str(avatar_path))
    else:
        image['images'][0].save(str(avatar_path))
