import os
import sys
from flick import Flick
from utils import *
from numpy import mean, var


if __name__ == "__main__":
    args = sys.argv
    if str(args[1]) == "exp1":
        begin_experiment_1(args[2])
    elif str(args[1]) == "exp2":
        if len(args) == 2:
            _x, _y = get_display_resolution()
            argl = [(13., _x, 0), (15., 2*_x, _y), (17., _x, 2*_y), (19., 0, _y)]
            str_list = []
            for (f, x, y) in argl:
                str_list.append("python2 src/run.py exp2 %f %i %i" % (f, x, y))
            str_list.append("python2 src/run.py classify")
            begin_experiment_2(str_list)

        elif len(args) == 3:
            render_waiting_screen()
            Flick(float(args[2])).flicker()
        else:
            Flick(float(args[2]), args[3], args[4]).flicker()
    elif str(args[1]) == "classify":
        start_live_classifier()
    else:
        print("Please specifiy exp1 or exp2 in the first argument!")
