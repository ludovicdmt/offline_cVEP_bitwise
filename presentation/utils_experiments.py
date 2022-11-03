import platform
import sys
import numpy as np
from psychopy import visual
import scipy.signal
from scipy.stats import pearsonr
import warnings

if platform.system() == "Linux":
    import re
    import subprocess
elif platform.system() == "Windows":
    import win32api
else:
    print("It looks like you are running this on a system that is not Windows or Linux. \
           Your system is not supported for running the experiment, sorry!")
    sys.exit()


class SingleFlicker:
    """Define and animate a single flicker using Pyschopy toolbox.

    The pattern of stimulation is pre-computed. More documentation
    to come, but some cleaning to be done before.

    Attributes
    ----------
    A lot, to be defined

    Methods
    -------
    Also a lot
    """

    def __init__(
        self,
        window,
        size=100,
        frequency=10,
        phase=0,
        amplitude=1.0,
        wave_type="sin",
        duration=2.2,
        fps=60,
        base_pos=(0, 0),
        rectang=False,
    ):
        self.window = window
        self.size = size
        self.total_size = self.size, self.size
        self.freq = frequency
        self.phase = phase
        self.amp = amplitude
        self.wave_type = wave_type
        self.rectang = rectang
        self.duration = duration
        self.fps = fps

        # Compute stimuli features
        self._initiate_wave()
        self.wave_func = self.set_wave_func()
        self.bound_pos = self._get_boundaries()
        self.base_x, self.base_y = self._check_base_pos(base_pos)
        self.flicker = self._make_flicker()

    def lfsr(self, taps, buf):
        """Function implements a linear feedback shift register
        Parameters
        ----------
        taps: list of int   
            List of Polynomial exponents for non-zero terms other than 1 and n

        buf: list of int
            List of buffer initialisation values as 1's and 0's or booleans

        Returns
        -------
        out: list of int
            The produced sequence

        """
        nbits = len(buf)
        sr = np.array(buf, dtype="bool")
        out = []
        for i in range((2**nbits) - 1):
            feedback = sr[0]
            out.append(feedback)
            for t in taps:
                feedback ^= sr[t]
            sr = np.roll(sr, -1)
            sr[-1] = feedback
        return out

    def make_gold(self, seq1, seq2):
        """Generate gold code from 2 m-seq"""
        # XOR
        gold = np.logical_xor(seq1, seq2)
        gold = np.array(gold, dtype=int)
        return gold

    def _initiate_wave(self):
        """For some wave function a pre-initialisation is required 
        before generating the actual wave that would be used.
        The pre-initialized wave is then adjusted, shifted or cut, 
        depending on the label number in the setter `wave_func()`.
        """
        length = int(self.duration * self.fps)

        if self.wave_type == "random":
            # Random sequence composed of subsequences of length 15 with 7 bit changes.
            # As in EEG2Code, codes pre-generated with code from author of the paper
            best_subset = np.load("randomEEG2Code.npy")
            sub_seq_per_code = int(length / 15) + 1

            cod = []
            tmp = []
            for idx, seq in enumerate(best_subset):
                tmp.extend(seq)
                if (idx + 1) % sub_seq_per_code == 0:
                    tmp = np.array(tmp)
                    tmp = tmp[:length]
                    if len(tmp) == length:
                        cod.append(tmp)
                    tmp = []
            if self.freq > len(cod):
                raise ZeroDivisionError(f"Maximum {len(cod)} frequencies.")

            self.wave = cod
        
        elif self.wave_type == 'shiftmseq':
            """Maximum length sequences https://en.wikipedia.org/wiki/Maximum_length_sequence
            """
            self.wave,_ = scipy.signal.max_len_seq(4, length=63)

        elif self.wave_type == "mseq":
            state = [
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [1, 0, 1, 1, 0, 1],
                [1, 0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
            ]
            taps = [[9], [3, 6, 9]]
            for s in state:
                s.extend([1, 0, 0, 1, 1])

            # Generate 10 codes of length 4094 samples
            cod = []
            for j in range(5):
                for i in range(2):
                    mseq = self.lfsr(buf=state[j], taps=taps[i])
                    mseq = np.repeat(mseq, 2)
                    cod.append(mseq)

            cod = np.array(cod)

            # In this bank of codes, we will slice shorter code that have
            # the proper length and that do not correlate to much with
            # the other ones and that have an approximately balanced
            # duty cyle

            alt_cod = []
            success = 0
            idx_cod = 0
            cod_number = 0

            # Exctract slices that match our criterion
            while (success < 44) and (idx_cod < 10):
                # Slice a first code
                tmp = cod[idx_cod][length * cod_number : length * (cod_number + 1)]

                # Check the duty cycle
                if (np.mean(tmp) > 0.45) and (np.mean(tmp) < 0.55):

                    # If it is not the first slice
                    if len(alt_cod) > 0:
                        max_cor = 0
                        # Check if the new slice correlate too much with
                        # the ones already in the bank of templates
                        for c in alt_cod:
                            corp, _ = pearsonr(tmp, c)
                            if corp > max_cor:
                                max_cor = corp
                        if np.abs(max_cor) < 0.5:
                            alt_cod.append(tmp)
                            success += 1
                    # If it is the first slice
                    else:
                        alt_cod.append(tmp)
                        success += 1

                cod_number += 1
                # When we have extracted all the possible slices in a
                # code, move to the following
                if cod_number > int(cod.shape[1] / length):
                    cod_number = 0
                    idx_cod += 1

            # Filter code that correlate too much with the first one
            # Reverse filtring compared to the one performed previously
            to_remove = []
            for j in range(len(alt_cod)):
                for i in range(j, len(alt_cod)):
                    if i == j:
                        continue
                    corp, _ = pearsonr(alt_cod[i], alt_cod[j])
                    if np.abs(corp) > 0.3:
                        to_remove.append(i)

            alt_cod = np.delete(alt_cod, to_remove, axis=0)
            self.wave = alt_cod.astype(int)
            
        elif self.wave_type == "code":
            # Chaotic codes from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6402685/pdf/pone.0213197.pdf
            A = 3.882
            x = 0.15
            wave = []
            for i in range(63):
                x = A * x * (1 - x)
                if x > 0.5:
                    wave.append(0)
                    wave.append(1)
                else:
                    wave.append(1)
                    wave.append(0)
            self.wave = np.array(wave)

        elif self.wave_type == "gold":
            if self.freq > 11:
                raise ZeroDivisionError("Maximum 11 frequencies.")
            state = [
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [1, 0, 1, 1, 0, 1],
                [1, 0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
            ]
            taps = [[5], [1, 4, 5]]

            tap1 = taps[0]
            tap2 = taps[1]

            idx_state1 = self.freq // 2
            state = state[idx_state1]

            # Generate two different m-seq
            seq1, _ = scipy.signal.max_len_seq(6, state=state, taps=tap1)
            seq1 = np.repeat(seq1, 3)
            seq2, _ = scipy.signal.max_len_seq(6, state=state, taps=tap2)
            seq2 = np.repeat(seq2, 3)

            # Merge the two to create a gold code
            wave = self.make_gold(seq1, seq2)

            # Keep only the corresponding length
            self.wave = wave[:length]

        elif self.wave_type == "random_slow":
            # Random sequence with 1 possible change every 7 bits.
            wave = []
            end = np.random.binomial(1, 0.5)
            start = np.abs(1 - end)
            seq = np.linspace(start, end, num=6)
            start = end
            wave.extend(seq)
            for _ in range(10):
                end = np.random.binomial(1, 0.5)
                seq = np.linspace(start, end, num=6)
                start = end
                wave.extend(seq)
            wave = np.array(wave)
            wave = wave[:63]

            wave = np.tile(wave, int(np.ceil(length / 63)))
            self.wave = wave[:length]

        else:
            raise ValueError('Sorry this type of wave is not implemented. Available waves are:\
                "sin", "square", "mseq", "shiftmseq", "code", "random", "gold", "randomslow".')

    def set_wave_func(self):
        """Genrerate the pattern of stimulation that will be use to animate the flicker"""
        length = int(self.duration * self.fps)

        if (self.wave_type == "sin") or (self.wave_type == "square"):
            if self.wave_type == "sin":
                wave_func = np.sin
            elif self.wave_type == "square":
                wave_func = scipy.signal.square
            frame_index = np.arange(0, length, 1)
            wave = 0.5 * (
                1
                + wave_func(
                    2 * np.pi * self.freq * (frame_index / self.fps) + (self.phase * np.pi)
                )
            )

        elif self.wave_type == "mseq":
            wave = self.wave[self.freq]

        elif self.wave_type == "shiftmseq":
            shift = int(self.freq * 63 / 11)
            wave = np.copy(self.wave)
            wave = np.roll(wave, shift)
            wave = np.tile(wave, int(np.ceil(length/63)))
            wave = wave[:length]

        elif self.wave_type == "code":
            shift = self.freq * int(63 / 11)
            wave = np.roll(self.wave, shift)
            wave = np.tile(wave, int(np.ceil(length / 63)))

        elif self.wave_type == "random":
            wave = self.wave[self.freq]

        elif (self.wave_type == "gold") or (self.wave_type == "random_slow"):
            wave = self.wave

        return wave

    def _get_boundaries(self):
        """Calculate the four boundaries of the screen based on the screen resolution"""

        # First, find the boundaries of the stim in the current resolution
        width, height = self.window.size

        # Create and populate the dict with the boundaries
        boundary_dict = {}

        boundary_dict["plus_x"] = (width // 2) - (self.size // 2) 
        boundary_dict["minus_x"] = (-width // 2) + (self.size // 2) 
        boundary_dict["plus_y"] = (height // 2) - (self.size // 2) 
        boundary_dict["minus_y"] = (-height // 2) + (self.size // 2) 

        return boundary_dict

    def _check_base_pos(self, base_pos):
        """Check that user's desired position is within boundaries, then return the position."""

        if type(base_pos) == tuple:
            # Unpack user-given position
            x_pos, y_pos = base_pos

            # Check the positions and give warnings (if any)
            if x_pos < self.bound_pos["minus_x"] or x_pos > self.bound_pos["plus_x"]:
                warnings.warn(
                    message=f"X position out of bounds. Valid values range from "
                    f"{self.bound_pos['minus_x']} and {self.bound_pos['plus_x']}"
                )

            if y_pos < self.bound_pos["minus_y"] or y_pos > self.bound_pos["plus_y"]:
                warnings.warn(
                    message=f"Y position out of bounds. Valid values range from "
                    f"{self.bound_pos['minus_y']} to {self.bound_pos['plus_y']}"
                )
        elif type(base_pos) == str:
            # Check for x-axis keywords
            if "left" in base_pos:
                x_pos = self.bound_pos["minus_x"]
            elif "right" in base_pos:
                x_pos = self.bound_pos["plus_x"]
            else:
                warnings.warn(
                    "Absent or invalid X-position string argument. Valid values are "
                    "'left' and 'right. Using 0 as X-position value..."
                )
                x_pos = 0

            if "down" in base_pos:
                y_pos = self.bound_pos["minus_y"]
            elif "up" in base_pos:
                y_pos = self.bound_pos["plus_y"]
            else:
                warnings.warn(
                    "Absent or invalid Y-position string argument. Valid values are "
                    "'up' and 'down. Using 0 as Y-position value..."
                )
                y_pos = 0

        else:
            warnings.warn(
                "Invalid position argument. It must be a tuple of integers or a "
                "correctly-formatted string. Using default position (0, 0)..."
            )

            x_pos, y_pos = (0, 0)

        return x_pos, y_pos

    def _make_flicker(self):
        """Create flicker.

        Position is calculated from the center
        """
        if self.rectang:
            rect = visual.Rect(
                self.window,
                width=self.size,
                height=self.size,
                pos=(self.base_x, self.base_y),
                lineColor="white",
                fillColor="white",
            )
        else:
            rect = visual.Circle(
                self.window,
                radius=self.size / 2,
                edges=50,
                pos=(self.base_x, self.base_y),
                lineColor="white",
                fillColor="white",
            )

        return rect

    def draw2(self, frame, amp_override=None):
        """Draw the flicker from a specific frame.

        Use a pre-calculated wave that only takes into
        account the frame index of the wave to decide on the opacity level.
        """

        # Get opacity for the flickers
        if amp_override is not None:
            opac_val = amp_override
        else:
            opac_val = self.wave_func[frame] * self.amp

        self.flicker.opacity = opac_val
        self.flicker.draw()


def get_screen_settings(platform):
    """
    Get screen resolution and refresh rate of the monitor

    Parameters
    ----------

    platform: str, ['Linux', 'Ubuntu']
              output of platform.system, determines the OS running this script

    Returns
    -------

    height: int
            Monitor height

    width: int
           Monitor width

    """

    if platform not in ["Linux", "Windows"]:
        print("Unsupported OS! How did you arrive here?")
        sys.exit()

    if platform == "Linux":
        cmd = ["xrandr"]
        cmd2 = ["grep", "*"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)

        p.stdout.close()

        info, _ = p2.communicate()
        screen_info = info.decode("utf-8").split()[:2]  # xrandr gives bytes, for some reason

        width, height = list(map(int, screen_info[0].split("x")))  # Convert to Int

    elif platform == "Windows":

        width, height = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)

    return width, height