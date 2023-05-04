import os

proj_dir = os.path.dirname(os.path.realpath(__file__))
vids_dir = os.path.join(proj_dir, "data", "videos")
raw_dir = os.path.join(vids_dir, "raw")
proc_dir = os.path.join(vids_dir, "processed")
mini_dir = os.path.join(vids_dir, "minicut")
model_dir = os.path.join(proj_dir, "data", "models")


colorlist = [
    "#00A8CC",
    "#2F80ED",
    "#4A90E2",
    "#4FC3F7",
    "#008080",
    "#00CED1",
    '#00FF00',
    '#80FF00',
    '#00FF80',
    '#008040',
    '#339933',
    '#FFA500',
    '#F50C2A',
    '#FF4D4D',
    '#D60A0A'
]
coord_keys = [
    'eye(back)',
    'eye(bottom)',
    'eye(front)',
    'eye(top)',
    'lowerlip',
    'mouth',
    'nose(bottom)',
    'nose(r)',
    'nose(tip)',
    'nose(top)',
    'nosebridge',
    'paw',
    'whisker(I)',
    'whisker(II)',
    'whisker(III)'
]
data_keys = ['likelihood', 'x', 'y']
