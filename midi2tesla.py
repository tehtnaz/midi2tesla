import numpy as np
import simpleaudio as sa

# high-level overview
# get midi parameters (tempo, PPQ, etc) -> merge all tracks into megatrack -> run generation to generate pulses -> postprocess -> export

# example usage: python midi2tesla.py --folder --reference_path . -c 16 "<midi>.mid"
# CLI parsing:
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("input")
parser.add_argument("-p", "--reference_path")
parser.add_argument("-o", "--output")
parser.add_argument("-f", "--folder", action="store_true")  # use input and output folders
parser.add_argument("-s", "--no_save_file", action="store_true")
parser.add_argument("-m", "--play_music", action="store_true")
parser.add_argument("-d", "--duty_cycle")

args = parser.parse_args()

SAMPLE_RATE = 44100  # 44100 samples per second

path = ""
if (args.reference_path != None):
    path = args.reference_path
midi = args.input
save_wav = midi  # output file name
if (args.output != None):
    save_wav = args.output
do_play_music = args.play_music  # play sound after conversion
in_path = path
out_path = path
if (args.folder):
    in_path = path + "/input/"
    out_path = path + "/output/"

midi_name = midi.split("/")[-1]
save_name = save_wav.split("/")[-1]

# more advanced playback settings
SAVE_FILE_TYPE = "wav"
A3_REF = 220  # A3 reference in case you want to change to a different reference frequency
do_save_wav = not args.no_save_file
PULSE_CORRECTION_FACTOR = 1  # affects pulse width. Usually pulse width is equal to note velocity in microseconds, but this can adjust that
SELECTED_TRACKS_IND = [-1]  # select indices of tracks to include; -1 indicates all tracks
MAX_PULSE_DURATION = 10  # in samples
MIN_PULSE_DURATION = 3
max_duty_cycle = 3  # maximum duty cycle per pulse, varied depending on note velocity
if (args.duty_cycle != None):
    max_duty_cycle = int(args.duty_cycle)
# some notes about pulse length and duty cycle. Lower duty cycle means more notes can be played without is ounding bad, but it reduces the apparent volume. Might be good to make it logarithmic since I don't thnk the correlation between volume and pulse width is linear
# there isn't really one setting that will work well for every song. Some songs require longer duty cycles and higher pulse widths to sound good, and others will just sound like noise without an extremely low duty cycle.

# tempo stuff
tempo = 500000  # initial tempo. Usually overridden unless tempoAutoset is False
TEMPO_AUTOSET = True


MAX_MIDI_IND = -1  # in case you only want a portion of the midi

# moving avg duty cycle limiter. Is this even needed? Its point is to reduce the chance of too much coil power draw, but that might be better implemented by a pulse width limiter.
MAX_DUTY = 0.5  # 50% duty cycle
MIN_DUTY = 0
WINDOW_SIZE = 1000

# deprecated
# maxpulselen=int((500/1000000)*fs) #max pulse len before turning off output
# timeoutlen=int((100/1000000)*fs) #after max pulse len exceeded, cooldown for this amount of samples
# has been replaced by moving average duty cycle-based limiting though it might be brought back as this is more relevant for Tesla coils

import soundfile as sf

def write_wav(data):
    sf.write(f"{out_path}{save_wav}.{SAVE_FILE_TYPE}", data, SAMPLE_RATE)


def ticks2samples(ticks):
    return int((ticks / ticks_per_beat) * (tempo / 1000000) * SAMPLE_RATE)  # fixed tempo issue


print(f"converting {midi} to {save_wav}.{SAVE_FILE_TYPE}")


class Tone:  # class containing tone generator and other tone playing information
    frequency = 0
    pulse_width = 0
    period = 0
    note = 0
    time = 0
    pitch = 0
    channel = 0
    velocity = 0

    def __init__(self, tone, velocity, channel):  # startTime will be timestamp to start #TODO: add pitch bend and continuous amplitude (pulse width) adjustment functions
        self.note = tone
        self.velocity = velocity
        self.channel = channel
        self.frequency = A3_REF * 2 ** ((tone - 69) / 12)  # ChatGPT'd
        self.set_freq(self.frequency)  # generate and calculate all the things

    # print(self.period)
    def change_pitch(self, new_pitch):  # for pitch bending functionality
        self.pitch = new_pitch
        bend_ratio = (new_pitch) / 4096  # allow 2 semitones up or down
        adjusted_note_number = self.note + bend_ratio
        self.frequency = A3_REF * (2 ** ((adjusted_note_number - 69) / 12))
        self.set_freq(self.frequency)
        return

    def set_freq(self, freq):
        self.period = int(((1 / freq) * SAMPLE_RATE) / 2)  # period in samples. Dividing by 2 works for some reason
        self.pulse_width = int((self.velocity / 127) * (max_duty_cycle / 100) * self.period)  # convert velocity to duty cycle (fraction of 1) then multiply by period to get the pulsewidth information
        # adding an offset/minimum on time can help balance out the relative strength of notes
        # should change this to make adjustment have a more continuous range
        if self.pulse_width > MAX_PULSE_DURATION:
            self.pulse_width = MAX_PULSE_DURATION
        if self.pulse_width < MIN_PULSE_DURATION:
            self.pulse_width = MIN_PULSE_DURATION
        self.pulse = np.concatenate((np.ones(self.pulse_width), np.zeros(self.period - self.pulse_width))).tolist()  # single pulse

    def generate(self, gen_time: int):  # generate for a certain amount of time (in samples)
        num_periods = int(gen_time / self.period) + 2  # the +2 is necessary to create a longer-then needed list that will be trimmed down
        pulses = self.pulse * num_periods  # generate a train of pulses
        init_time = self.time % self.period  # time in pulsetrain to start at (which is the index in a single pulse the last pulse started at)
        final = np.array(pulses[init_time : gen_time + init_time])  # slice to get a list with a length of gentime elements and turn into numpy array
        self.time += gen_time  # update self.time
        return final


tones: list[Tone] = []

def find_in_tones(note):
    for ind in range(len(tones)):
        if tones[ind].note == note:  # could be sped up
            return ind


def play_music(music):
    dt = time.monotonic()
    audio = music * (2**15 - 1)
    # Convert to 16-bit data
    audio = audio.astype(np.int16)

    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, SAMPLE_RATE)

    # Wait for playback to finish before exiting
    play_obj.wait_done()
    print(f"song done in {time.monotonic()-dt} seconds")


from mido import Message
import mido

mid = mido.MidiFile(f"{in_path}{midi}")
ticks_per_beat = mid.ticks_per_beat  # THIS FIXES ALL THE TEMPO PROBLEMS AHHHHHH
# print(f"ticks per beat: {ticksPerBeat}")
tempo_set = False  # whether tempo has been set
for track in mid.tracks:
    for msg in track[: min((len(track), 10))]:
        # print(msg)
        if msg.type == "set_tempo":
            if TEMPO_AUTOSET and not tempo_set:
                tempo_set = True
                tempo = msg.tempo  # comment out if tempo is completely wrong
                print(f"found tempo: {mido.tempo2bpm(msg.tempo)} bpm")


print("converting relative times to absolute times and merging tracks")
import time

dt = time.monotonic()
biggest_track = max(mid.tracks, key=len)
curr_samp_time = 0
# preprocessing midi to convert all tracks into one

selected_tracks: list[list[Message]] = []
if SELECTED_TRACKS_IND == []:
    selected_tracks.append(biggest_track)
elif SELECTED_TRACKS_IND == [-1]:
    selected_tracks = mid.tracks
else:
    for track_ind in SELECTED_TRACKS_IND:
        selected_tracks.append(mid.tracks[track_ind])

# merges all notes into one list, and converts each note to use abs time rather than relative time
def get_mega_track():
    abs_tick_times = np.zeros(len(selected_tracks)).tolist()  # stores the absolute (not relative) tick value for each track
    mega_track: list[Message] = []
    biggest_len = len(biggest_track)
    ind = 0
    while ind <= biggest_len:
        for track_ind, track in enumerate(selected_tracks):
            if (ind + 1) <= len(track):
                # do some list manipulation
                abs_tick_times[track_ind] += track[ind].time
                new_obj = track[ind]
                new_obj.time = abs_tick_times[track_ind]
                mega_track.append(new_obj)
        ind += 1
    return mega_track

mega_track = sorted(get_mega_track(), key=lambda x: x.time)
# this midi object now has times stored as absolute times not relative times.


# tones=[None]*20 #support max 20 note polyphony. Should be
total_samples = int(ticks2samples(mega_track[-1].time) + WINDOW_SIZE)  # sum total number of samples
music = np.zeros(total_samples)  # placeholder array to be filled by samples. Much more efficient than np.append()
abs_time = 0
biggest_len = len(mega_track)
for ind, msg in enumerate(mega_track[:MAX_MIDI_IND]):  # will add multiple tracks later
    if msg.type == "note_on" or msg.type == "note_off" or msg.type == "pitchwheel":
        if ind % 100 == 0:
            print(f"processed {ind}/{biggest_len} commands")
            pass
        gen_time = ticks2samples(msg.time - abs_time)  # convert to samples WARNING: midi time is in absolute time rather than relative, so it will not process normal midi that has not run through preprocessing without modification
        music[curr_samp_time : curr_samp_time + gen_time] = np.where(sum([tone.generate(gen_time) for tone in tones]) > 0, 1, 0)  # maybe could cut down on compute by replacing sum with something
        # for more advanced polyphony, modify the sum() and tone.generate (probably mess with the offset time)
        curr_samp_time += gen_time
        abs_time += msg.time - abs_time
        if msg.type == "note_on":
            # pulseDuration = msg.velocity*pulseCorrectionFactor TODO: possibly add some sort of compensation for the higher and lower frequencies
            tones.append(Tone(msg.note, msg.velocity, msg.channel))  # append note to current note playing
        if msg.type == "note_off":
            tone_to_remove = find_in_tones(msg.note)
            if tone_to_remove is not None:
                tones.pop(tone_to_remove)  # remove note once it stops playing
    if msg.type == "tempo":
        tempo = msg.tempo
    if msg.type == "pitchwheel":
        for ind in range(len(tones)):
            if tones[ind].channel == msg.channel:
                tones[ind].change_pitch(msg.pitch)
        # find all tones with same channel attribute and change pitch


# TODO: Drum implementation using a single long pulse
print(f"postprocessing {len(music)} elements to limit duty cycle to {MAX_DUTY*100}%: ")


# postprocess
# window=np.zeros(windowsize)
def moving_avg(data, window):  # ChatGPT'd
    # Create a simple moving average kernel
    kernel = np.ones(window) / window
    # Use np.convolve to compute the moving average
    return np.convolve(data, kernel, mode="valid")


avg = moving_avg(music, WINDOW_SIZE)
music = np.where((avg >= MAX_DUTY) | (avg <= MIN_DUTY), 0, music[: -WINDOW_SIZE + 1])
print("postprocessing complete")
print()

print(f"conversion complete: {len(music)/SAMPLE_RATE} seconds of music ({len(music)} samples, {len(mid.tracks) - 1} tracks)")
print(f"with tempo: {tempo} us/beat")
print(f"completed in {time.monotonic()-dt} seconds.")
if do_save_wav:
    print(f"saving file as {save_wav}.{SAVE_FILE_TYPE}")
    write_wav(music)
if do_play_music:
    print(f"playing music...")
    play_music(music)
