SAMPLE_RATE = 16000
HOP_SIZE = 256
VOICED_THRESHOLD = 0.5

SEED = 1
OUT = "runs"
DEVICE = "auto"
WORKERS = 8

EPOCHS = 270
WARMUP_EPOCHS = 5
SWA_TAIL = 0.2
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 1e-05
MIN_LR_FACTOR = 1e-06
LR_DECAY_FRAC = 0.2
CHUNK_SECONDS = 0.5
CORRECTNESS_LOSS_WEIGHT = 1.0
CORRECTNESS_TOL_CENTS = 50.0

PATTERNS = [
    {"name": "identity", "stages": []},
    {"name": "scene", "stages": ["scene", "level"]},
    {"name": "mic", "stages": ["mic", "level"]},
    {"name": "scene_mic", "stages": ["scene", "mic", "level"]},
    {"name": "full", "stages": ["scene", "room", "mic", "level"]},
]

CTX_SECONDS = 2.0
MARGIN_SECONDS = 0.256
SOURCE_SLOTS = 2
NOISE_ALPHA_RANGE = (0.0, 1.0)
NOISE_FMIN_HZ = 20.0
SIR_DB_RANGE = (-6.0, 15.0)
VOICELESS_RMS_DBFS_RANGE = (-40.0, -10.0)
LEVEL_PEAK_DBFS_RANGE = (-35.0, -5.0)
F_LO_CHOICES = (20.0, 300.0)
F_HI_HZ_LOG_RANGE = (2000.0, 8000.0)
MIC_HALF_ORDER_CHOICES = (2, 4, 8)
BRICKWALL_PROB = 0.5
BRICKWALL_EDGE_HZ = 40.0
MP3_PROB = 0.5

BN_RECAL_BATCHES = 100
CALIB_CLIPS_PER_CORPUS = 150
CALIB_MAX_SECONDS = 10.0

# Audio sampled from the training corpora for the export checks.
ONNX_CHECK_SEED = 7103
ONNX_CHECK_CLIPS_PER_CORPUS = 6
ONNX_CHECK_MAX_SECONDS = 10.0

CORPORA = [
    ("MDBStemSynth", 0.35),
    ("NSynth", 0.15),
    ("PTDB", 0.1667),
    ("MOCHA", 0.1667),
    ("CMUArctic", 0.1667),
]
POWER_FLOOR = 1e-12
RMS_FLOOR = 1e-9
