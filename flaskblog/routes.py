import os
import time
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort
from flaskblog import app, db, bcrypt
from flaskblog.forms import RegistrationForm, LoginForm, UpdateAccountForm, PostForm, LandingForm, ChordForm
from flaskblog.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
from flask_uploads import UploadSet, AUDIO

####################################### Deep Learning Imports ##########################################
import numpy as np
import shutil
import tensorflow as tf
import magenta.music as mm
from magenta.music.sequences_lib import concatenate_sequences
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import ctypes.util
def proxy_find_library(lib):
  if lib == 'fluidsynth':
    return 'libfluidsynth.so.1'
  else:
    return ctypes.util.find_library(lib)
ctypes.util.find_library = proxy_find_library

############################################## Definitions #################################################
BATCH_SIZE = 4
Z_SIZE = 512
TOTAL_STEPS = 512
BAR_SECONDS = 2.0
CHORD_DEPTH = 49

SAMPLE_RATE = 44100
SF2_PATH = 'content/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'

# Play sequence using SoundFont.
def play(note_sequences):
  if not isinstance(note_sequences, list):
    note_sequences = [note_sequences]
  for ns in note_sequences:
    mm.play_sequence(ns, synth=mm.fluidsynth, sf2_path=SF2_PATH)
  
# Spherical linear interpolation.
def slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)), np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


# Chord encoding tensor.
def chord_encoding(chord):
  index = mm.TriadChordOneHotEncoding().encode_event(chord)
  c = np.zeros([TOTAL_STEPS, CHORD_DEPTH])
  c[0,0] = 1.0
  c[1:,index] = 1.0
  return c

# Trim sequences to exactly one bar.
def trim_sequences(seqs, num_seconds=BAR_SECONDS):
  for i in range(len(seqs)):
    seqs[i] = mm.extract_subsequence(seqs[i], 0.0, num_seconds)
    seqs[i].total_time = num_seconds

# Consolidate instrument numbers by MIDI program.
def fix_instruments_for_concatenation(note_sequences):
  instruments = {}
  for i in range(len(note_sequences)):
    for note in note_sequences[i].notes:
      if not note.is_drum:
        if note.program not in instruments:
          if len(instruments) >= 8:
            instruments[note.program] = len(instruments) + 2
          else:
            instruments[note.program] = len(instruments) + 1
        note.instrument = instruments[note.program]
      else:
        note.instrument = 9

######################################### ROUTES ############################################
@app.route("/")
@app.route("/home")
def home():
    if current_user.is_authenticated:
        return redirect(url_for('landing'))
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('landing'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)


@app.route("/landing", methods=["GET","POST"])
@login_required
def landing():
    form = LandingForm()
    if form.validate_on_submit():
        num_bars = form.num_bars.data
        temperature = form.temperature.data
        config = configs.CONFIG_MAP['hier-multiperf_vel_1bar_med']
        weight_name = 'model_fb256.ckpt'
        model = TrainedModel(config, batch_size=BATCH_SIZE,
                checkpoint_dir_or_path=os.path.join(app.root_path, 'content', weight_name))
        model._config.data_converter._max_tensors_per_input = None

        z1 = np.random.normal(size=[Z_SIZE])
        z2 = np.random.normal(size=[Z_SIZE])
        z = np.array([slerp(z1, z2, t)
                      for t in np.linspace(0, 1, num_bars)])

        seqs = model.decode(length=TOTAL_STEPS, z=z, temperature=temperature)

        trim_sequences(seqs)
        fix_instruments_for_concatenation(seqs)
        interp_ns = concatenate_sequences(seqs)
        f_ext = '.mid'
        random_hex = secrets.token_hex(8)
        music_fn = random_hex + f_ext
        music_mp3 = random_hex+".mp3"
        #Save music to disk
        mm.sequence_proto_to_midi_file(interp_ns, music_fn)

        ### Move audio to a specified path
        source_path = "/home/pratheesh/Flask_Blog/" + music_mp3
        destination_path = os.path.join(app.root_path, 'static/music', music_mp3)
        
        cmd_to_wav = "fluidsynth -F "+ random_hex +".wav /usr/share/sounds/sf2/FluidR3_GM.sf2 "+ music_fn
        print(cmd_to_wav)
        os.system(cmd_to_wav)
        cmd_to_mp3 = "lame --preset standard "+ random_hex + ".wav " + random_hex +".mp3"
        print(cmd_to_mp3)
        os.system(cmd_to_mp3)
        #shutil.move(source_path, destination_path)
        #os.replace(source_path, destination_path)
        print("moving file")
        os.rename(source_path, destination_path)
        os.remove(music_fn)
        os.remove(random_hex + ".wav")
        
        music_file = url_for('static', filename='music/' + music_mp3)


        return render_template('music_output.html', music_file=music_file)

    return render_template('landing.html', form=form)

@app.route("/gen_chords", methods=["GET","POST"])
@login_required
def gen_chords():
    form = ChordForm()
    print("Chords CHOICES from form:" + str(form.chord_1.choices))
    form.chord_2.choices = [(chord_id, chord_name) for chord_id, chord_name in form.chord_1.choices]
    form.chord_3.choices = [(chord_id, chord_name) for chord_id, chord_name in form.chord_1.choices]
    form.chord_4.choices = [(chord_id, chord_name) for chord_id, chord_name in form.chord_1.choices]
    if form.validate_on_submit():
        chord_1 = form.chord_1.data
        chord_2 = form.chord_2.data
        chord_3 = form.chord_3.data
        chord_4 = form.chord_4.data
        chords = [chord_1, chord_2, chord_3, chord_4]

        num_bars = form.num_bars.data
        temperature = form.temperature.data

        config = configs.CONFIG_MAP['hier-multiperf_vel_1bar_med_chords']
        weight_name = 'model_chords_fb64.ckpt'
        model = TrainedModel(config, batch_size=BATCH_SIZE,
                checkpoint_dir_or_path=os.path.join(app.root_path, 'content', weight_name))

        z1 = np.random.normal(size=[Z_SIZE])
        z2 = np.random.normal(size=[Z_SIZE])
        z = np.array([slerp(z1, z2, t)
                      for t in np.linspace(0, 1, num_bars)])

        seqs = [
            model.decode(length=TOTAL_STEPS, z=z[i:i+1, :], temperature=temperature,
                         c_input=chord_encoding(chords[i % 4]))[0]
            for i in range(num_bars)
        ]

        trim_sequences(seqs)
        fix_instruments_for_concatenation(seqs)
        prog_interp_ns = concatenate_sequences(seqs)

        f_ext = '.mid'
        random_hex = secrets.token_hex(8)
        music_fn = random_hex + f_ext
        music_mp3 = random_hex+".mp3"

        #Save music to disk
        mm.sequence_proto_to_midi_file(prog_interp_ns, music_fn)

        ### Move audio to a specified path
        source_path = "/home/pratheesh/Flask_Blog/" + music_mp3
        destination_path = os.path.join(app.root_path, 'static/music', music_mp3)
        
        cmd_to_wav = "fluidsynth -F "+ random_hex +".wav /usr/share/sounds/sf2/FluidR3_GM.sf2 "+ music_fn
        print(cmd_to_wav)
        os.system(cmd_to_wav)
        cmd_to_mp3 = "lame --preset standard "+ random_hex + ".wav " + random_hex +".mp3"
        print(cmd_to_mp3)
        os.system(cmd_to_mp3)
        #shutil.move(source_path, destination_path)
        #os.replace(source_path, destination_path)
        print("moving file")
        os.rename(source_path, destination_path)
        os.remove(music_fn)
        os.remove(random_hex + ".wav")
        
        music_file = url_for('static', filename='music/' + music_mp3)


        return render_template('music_output.html', music_file=music_file)

    return render_template('gen_chords.html', form=form)