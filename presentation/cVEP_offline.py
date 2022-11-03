"""
T9 type numerical keyboard using flickers for BCI-demonstration

Authors: Juan Jesus Torre Tresols and Ludovic Darmet and Giuseppe Ferraro
email: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import json
import os
import platform
import random
import numpy as np

import argparse
import time

from psychopy import visual, core, event
from pylsl import StreamInfo, StreamOutlet
from utils_experiments import get_screen_settings, SingleFlicker


def pause():
    """Pause execution until the 'c' key is pressed"""

    paused = True

    while paused:
        if event.getKeys('space'):
            paused = False
        elif event.getKeys('s'):
            return "Skip"


# Load config file
path = os.getcwd()


parser = argparse.ArgumentParser(description='Config file name')
parser.add_argument('-f', '--file', metavar='ConfigFile', type=str,
                    default='T9_config_cVEpoffline.json', help="Name of the config file for freq "
                    "and amplitude. Default: %(default)s.")

args = parser.parse_args()
config_path = os.path.join(path, args.file)

with open(config_path, 'r') as config_file:
    params = json.load(config_file)

# Experimental params
size = params['size']
trial_n = params['trial_n']
cal_n = params['cal_n']
epoch_duration = params['epoch_duration']
iti_duration = params['iti_duration']
cue_duration = params['cue_duration']

# Stim params

# JSON does not like tuples...
positions = [tuple(position) for position in params['positions']]
number_codes = params['flicker_codes']
amp = params['amplitude']

# Marker stream
info = StreamInfo(name='MyMarkerStream', type='Markers', channel_count=1,
                  nominal_srate=0, channel_format='string', source_id='myuidw43536')
info.desc().append_child_value("n_train", f"{cal_n}")
info.desc().append_child_value("n_class", f"{len(number_codes)}")
info.desc().append_child_value("amp", str(amp))


# Window parameters
system = platform.system()
width, height = get_screen_settings(system)

# Background:
color = 'darkgray'
wave_type = 'shiftmseq'

b = (-1., -1., -1.)

if color == 'black':
    c = [-1.000, -1.000, -1.000]
elif color == 'darkgray':
    c = [-.500, -.500, -.500]
else:  # Make it gray
    c = [0, 0, 0]
window = visual.Window([width, height], screen=1, color=c, waitBlanking=False, 
numSamples=1, units="pix", fullscr=True)
window.mouseVisible = False

refresh_rate = round(window.getActualFrameRate())
print('Refresh rate', refresh_rate)
# Time conversion to frames
epoch_frames = int(epoch_duration * refresh_rate)
if len(number_codes) == 1:
    iti_duration /= iti_duration
iti_frames_cal =  int(iti_duration * refresh_rate)
cue_frames = int(cue_duration * refresh_rate)

# Stim
calib_text_start = "Start the cVEP stimulation. \n \n \
 Please press space when you are ready."
calib_text_end = "Experiment ended! Thank you for your participation. Press 'Esc' or 'Q' to exit..."

trial_text = "Code entered correctly. Press 'C' to start a new trial..."
exp_text = "Experiment ended! Thank you for your participation. Press 'Esc' or 'Q' to exit..."

cal_start = visual.TextStim(window, text=calib_text_start)
cal_end = visual.TextStim(window, text=calib_text_end)
trial_end = visual.TextStim(window, text=trial_text)
exp_end = visual.TextStim(window, text=exp_text)

cue_size = size + 5
cue = visual.Rect(window, width=cue_size, height=cue_size,
                  pos=[0, 0], lineWidth=10, fillColor =  None, lineColor='red')

buttons = {f"{code}": visual.TextStim(win=window, text=code, pos=pos, color=b, height=35)
           for pos, code in zip(positions, number_codes)}

flickers = {f"{code}": SingleFlicker(window=window, size=size, frequency=freq, phase=phase, amplitude=amp,
                                    wave_type=wave_type, duration=epoch_duration, fps=refresh_rate, base_pos=pos)
            for freq, pos, phase, code in zip(range(len(number_codes)), positions, range(len(number_codes)), number_codes)}
codes = []
for f in flickers.values():
    c = str(f.wave_func)
    c = c.replace('\n', '')
    c = c.replace('[', '')
    c = c.replace(']', '')
    c = c.replace(' ', '')
    codes.append(c)
codes = np.array(codes)

classes = []
for idx, f in enumerate(codes):
    classes.append(str(f) + '_' + '_' + str(number_codes[idx]))
info.desc().append_child_value("events_labels", ','.join(classes))
outlet = StreamOutlet(info)

# Experiment structure
# Cue sequence for each block
trial_list = []
symbols = number_codes
for _ in range(cal_n):
    sequence = random.sample(symbols, len(symbols))
    trial_list.append(sequence)

# Presentation
cal_start.draw()
window.flip()
out = pause()


if out != "Skip":

    for idx_block, sequence in enumerate(trial_list):

        # Draw the number codes
        for button in buttons.values():
            button.autoDraw = False

        if len(number_codes) > 1:
            txt = f'Block {idx_block+1} out of {len(trial_list)}. \n Please press space to continue.'
            visual.TextStim(window, text=txt).draw()
            window.flip()
            pause()

        # Draw the number codes
        for button in buttons.values():
            button.autoDraw = True

        # For each number in our sequence...
        shift_idx = 0
        for target in sequence:
            # Select target flicker
            target_flicker = flickers[str(target)]
            target_pos = (target_flicker.base_x, target_flicker.base_y)
            target_freq = target_flicker.freq
            target_phase = target_flicker.phase
            code = target_flicker.wave_func
            code = str(code)
            code = code.replace('\n', '')
            code = code.replace('[', '')
            code = code.replace(']', '')
            code = code.replace(' ', '')

            # ITI presentation
            for n in range(iti_frames_cal):
                for flicker in flickers.values():
                    flicker.draw2(frame=0, amp_override=1.0)
                window.flip()

            # Cue presentation
            cue.pos = target_pos
            for frame in range(cue_frames):
                for flicker in flickers.values():
                    flicker.draw2(frame=0, amp_override=1.0)
                # Draw the cue over the static flickers
                cue.draw()
                window.flip()

            # Flicker presentation
            marker_info = [f"{code}_{target}"]
            outlet.push_sample(marker_info)
            frames = 0
            t0 = time.time()  # Retrieve time at trial onset

            for frame, n in enumerate(range(epoch_frames)):
                for flicker in flickers.values():
                    flicker.draw2(frame=frame)
                frames += 1
                window.flip()

            t1 = time.time()  # Time at end of trial
            elapsed = t1 - t0
            print(f"Time elapsed: {elapsed}")
            print(f"Total frames: {frames}")
            print("")

            # Make them transparent
            for flicker in flickers.values():
                flicker.draw2(0, amp_override=0) 
            window.flip()            

    for button in buttons.values():
        button.autoDraw = False
    cal_end.draw()
    window.flip()
    pause()
