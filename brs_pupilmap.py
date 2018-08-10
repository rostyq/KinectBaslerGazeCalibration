from cv2 import waitKey
from cv2 import imshow
from cv2 import imread
from cv2 import namedWindow
from cv2 import setMouseCallback
from cv2 import destroyAllWindows
from cv2 import EVENT_LBUTTONUP
from cv2 import resize
from cv2 import cvtColor
from cv2 import COLOR_RGB2GRAY
from cv2 import equalizeHist

import json
import os
import sys
sys.path.append('C:/Users/Beehiveor/PycharmProjects/')

from brspy.export import GazesExport
from brspy.export import MimicsExport
from brspy.export import JointsExport
from brspy.export import JointOrientationsExport
from brspy.export import FacePropertiesExport

from brspy.reader import Session

__name__ = 'Pupil Center Labeling Tool'
ESC_KEY = 27
ix, iy = -1, -1
read_next = True
factor = 4

# helping fuctions 
def to_tuple(dct):
    return tuple(int(item) for key, item in dct.items())

def extract_rectangle(img, pt0, pt1):
    return img[pt0[1]:pt1[1], pt0[0]:pt1[0]]

def ispressed(button, delay=1):
    return True if waitKey(delay) == button else False

def read_coordinates(event, x, y, flags, param):
    """
    Mouse callback function
    """
    global ix, iy
    global read_next

    if event == EVENT_LBUTTONUP:
        ix, iy = x, y
        read_next = True

session_path = sys.argv[1]
print(f'READING: {session_path}\n\r')
sess = Session(session_path)

avoid = [
    'KinectDepth',
    'KinectInfrared',
    'KinectBodyIndex',
    'KinectColor',
    'KinectBody',
    'KinectFaceVertices',
#    'GazeEstimation',
    'WebCamera',
#    'InfraredCamera',
    'Markers',
    'ManualPupils'
]
sess.remove_devices(*avoid)

namedWindow(__name__)
setMouseCallback(__name__, read_coordinates)

# iterate on snapshots
for snapshot in sess.snapshots_iterate(1500):
    
    data = {'pupilCenterLeft': {}, 'pupilCenterRight': {}}

    # iterate on left and right eye
    for eye in ['Left', 'Right']:

        # debug
        # print(snapshot.GazeEstimation[f'eyeRoi{eye}'])
        
        # get roi points
        roi = [to_tuple(pt) for pt in snapshot.GazeEstimation[f'eyeRoi{eye}']]
        
        # get eye image
        eye_img = cvtColor(extract_rectangle(snapshot.InfraredCamera, *roi), COLOR_RGB2GRAY)
        
        eye_img = equalizeHist(resize(eye_img, None, fx=factor, fy=factor))

        read_next = False

        # show image event
        while not ispressed(ESC_KEY):    
            if read_next:
                break
            imshow(__name__, eye_img)
       
        # rescale coordinates and add left upper corner of roi
        X = roi[0][0] + ix / factor
        Y = roi[0][1] + iy / factor

        data[f'pupilCenter{eye}'] = {'X': X, 'Y': Y}

    filename = str(snapshot.number).rjust(5, '0') + '.txt'
    
    # write data
    pupil_labels_path = os.path.join(sess.path, 'DataSource', 'cam_102')
    try:
        os.mkdir(pupil_labels_path)
    except:
        pass

    with open(os.path.join(pupil_labels_path, filename), 'w') as f:
        json.dump(data, f)

destroyAllWindows()
