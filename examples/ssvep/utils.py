from screeninfo import get_monitors
import pygame
from pygame.locals import *
import os
import sys
from flick import Flick
import time
from record_data import RecordData
from live_recorder import LiveRecorder
from sklearn.externals import joblib
import numpy as np
from preprocess import preprocess_recordings
from subprocess import Popen


pygame.init()


def get_display_resolution():
    """
    | Returns half of width and height of screen in pixels
    """
    h_str = str(get_monitors()[0])
    for char in ['+', '(', ')', 'x']:
        h_str = h_str.replace(char, '|')
    w, h = (h_str.split('|')[1], h_str.split('|')[2])
    return (int(w)/2, int(h)/2)


def time_str():
    return time.strftime("%H_%M_%d_%m_%Y", time.gmtime())


def render_waiting_screen(text_string=None, time_black = 0.):
    pygame.font.init()
    display_x, display_y = get_display_resolution()
    display_x, display_y = (2 * display_x, 2 * display_y)
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    window = pygame.display.set_mode((display_x, display_y), pygame.NOFRAME, 32)
    pygame.display.set_caption("SSVEP")
    if time_black > 0:
        window.fill((0., 0., 0.))
        timer_event = USEREVENT + 1
        pygame.time.set_timer(timer_event, int(time_black)*1000)
    else:
        myfont = pygame.font.SysFont("arial", 50)
        press_string = "Please press the Any-Key to continue..."
        textsurface1 = myfont.render(press_string, False, (0, 0, 0))
        text_rect1 = textsurface1.get_rect(center=(display_x/2, display_y/2+100))
        if text_string:
            textsurface2 = myfont.render(text_string, False, (0, 0, 0))
            text_rect2 = textsurface2.get_rect(center=(display_x/2, display_y/2-100))
        window.fill((100, 100, 150))
        window.blit(textsurface1, text_rect1)
        if text_string:
            window.blit(textsurface2, text_rect2)
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    exit()
                else:
                    pygame.quit()
                    return False
            if not (time_black > 0.):
                window.blit(textsurface1, text_rect1)
                if text_string:
                    window.blit(textsurface2, text_rect2)
            else:
                if event.type == timer_event:
                    pygame.quit()
                    return False
            pygame.display.update()


def begin_experiment_1(freq, trials=20):
    if not os.path.isdir("REC"):
        os.mkdir("REC")
    render_waiting_screen("Welcome to this experiment")
    render_waiting_screen("The experiment will start now... there will be breaks between the flickering tiles!")
    recorder = RecordData(256., 20., freq)
    recorder.start_recording()

    for i in range(0, int(trials)):
        recorder.add_trial(int(freq))
        Flick(float(freq)).flicker(15.)
        recorder.add_trial(0.)
        render_waiting_screen(text_string=None, time_black=5.)

    filename = "REC/%s_freq_%s.mat" % (time_str(), freq)
    recorder.stop_recording_and_dump(filename)
    recorder.killswitch.terminate = True
    recorder = None

    render_waiting_screen("That was the last one, thank you for participation!")
    sys.exit()


def begin_experiment_2(str_list):
    display_x, display_y = get_display_resolution()
    display_x, display_y = (2 * display_x, 2 * display_y)
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    window = pygame.display.set_mode((display_x, display_y), pygame.NOFRAME, 32)
    pygame.display.set_caption("SSVEP")
    window.fill((0, 0, 0))
    pygame.display.update()

    if os.name == 'nt':
    	for command in str_list:
            command_parts = command.split(" ")
            #print("start /d "+command)
            #os.system("start /d "+command)
            Popen(command_parts)
    elif os.name == 'posix':
        os.system("|".join(str_list))
    else:
        print("Could not get OS-name!")


def start_live_classifier():
    window_metrics = (200, 200)
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    window = pygame.display.set_mode(window_metrics, pygame.NOFRAME, 0)
    pygame.display.set_caption("classifier window")
    pygame.mouse.set_visible(False)
    arrow = pygame.transform.scale(pygame.image.load("src/res/arrow.png"), window_metrics)
    stop = pygame.transform.scale(pygame.image.load("src/res/stop.png"), window_metrics)
    arrow_metrics = window_metrics
    window.blit(stop, (0, 0))
    pygame.display.update()
    # Start Recording
    recorder = LiveRecorder()
    recorder.start_recording()
    time.sleep(1)

    #labels, features = getData(np.load('19_06_05_07_2017_freq_19.mat.npy'))

    do_run = True
    model_file_QDA, model_file_LDA, model_file_MLP = ('src/QDA.pkl', 'src/LDA.pkl', 'src/MLP.pkl')

    QDA = joblib.load(model_file_QDA)
    LDA = joblib.load(model_file_LDA)
    MLP = joblib.load(model_file_MLP)

    label = None
    time.sleep(5)

    label_list = []

    while do_run:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN or label:
                try:
                    a = event.key
                    label = None
                except AttributeError:
                    event.key = None
                if event.key == K_ESCAPE:
                    do_run = False
                elif event.key == K_UP or label == 13.0:    # UP is 13 Hz
                    window.fill((0., 0., 0.))
                    window.blit(rot_center(arrow, 180), (0, 0))
                    # TODO move robot up
                elif event.key == K_DOWN or label == 17.0:  # DOWN is 17 Hz
                    window.fill((0., 0., 0.))
                    window.blit(arrow, (0, 0))
                    # TODO move robot down
                elif event.key == K_RIGHT or label == 15.0:    # RIGHT is 15 Hz
                    window.fill((0., 0., 0.))
                    window.blit(rot_center(arrow, 90), (0, 0))
                    # TODO move robot right
                elif event.key == K_LEFT or label == 19.0:   # LEFT is 19 Hz:
                    window.fill((0., 0., 0.))
                    window.blit(rot_center(arrow, 270), (0, 0))
                    # TODO move robot left
                elif event.key == K_SPACE or label == 0.0:  # No frequency
                    window.fill((0., 0., 0.))
                    window.blit(stop, (0, 0))
                    # TODO stop robot
                label = None

            elif event.type == KEYUP:
                window.fill((0., 0., 0.))
                window.blit(stop, (0, 0))
                # TODO stop robot

            pygame.display.flip()

        features = recorder.get_features()
        #print(features)
        label_LDA = LDA.predict([features])[0]
        label_QDA = QDA.predict([features])[0]
        label_MLP = MLP.predict([features])[0]
        print("LDA: %s QDA: %s MLP: %s" %(label_LDA, label_QDA, label_MLP))
        for tmp_label in [label_LDA, label_QDA, label_MLP]:
            label_list.append(tmp_label)

        if len(label_list) >= 10*3:
            count_list = [label_list.count(13.), label_list.count(15.)]
            count_list.append(label_list.count(17.))
            count_list.append(label_list.count(19.))
            index = np.argmax(count_list)
            label = [13., 15., 17., 19.][index]
            print("Mayor Label: %s" % label)
            label_list = []

        #print("Recognized Label: %s" % label)
        time.sleep(1.)

    #   May dump labeled data after recording?
    filename = "REC/live_%s_freq_%s.mat" % (time_str(), freq)


def rot_center(image, angle):
    """rotate an image while keeping its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image

if __name__ == "__main__":
    begin_experiment_1()
    exit()
