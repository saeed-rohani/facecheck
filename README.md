# facecheck

Register people with your webcam, then check who is in front of it.
`facecheck` saves each person's face data and measurements in a local
database and matches new faces against it. It needs only OpenCV and NumPy,
and pip installs both for you.

```console
$ pip install facecheck
$ facecheck register "Saeed Rohani"
$ facecheck check
```

Everything runs on your own computer. No image is ever saved or uploaded:
only face data (numbers) is stored, in a database on this machine.

## What it does

- **`facecheck register NAME`** opens the camera, waits until the person
  looks straight at it, takes 5 face samples, measures the face, and saves it
  all under that name.
- **`facecheck check`** opens the camera and says who it is: the best match
  among everyone registered, or "no match".
- **`facecheck check --name NAME`** answers a narrower question: is this
  NAME?

Under the hood, all three models come from OpenCV:

| Step              | Model                                     | Size         |
| ----------------- | ----------------------------------------- | ------------ |
| Find the face     | YuNet                                     | about 0.2 MB |
| Tell people apart | SFace, a 128-number "face print" per face | about 37 MB  |
| Measure the face  | Facemark LBF, 68 landmarks                | about 54 MB  |

The models download automatically the first time a command needs them and
are cached; after that no network is needed.

## Install

```console
$ pip install facecheck
```

That is all: pip also installs the two dependencies, `opencv-contrib-python`
and `numpy`. To be safe on older computers, you can add `--only-binary`, which
makes pip use ready-made builds instead of trying to compile OpenCV:

```console
$ pip install --only-binary=opencv-contrib-python,numpy facecheck
```

Requirements:

- Python 3.10 or newer
- A webcam
- Windows (64- or 32-bit); macOS 11+ on Apple silicon or 12+ on Intel; or
  Linux with glibc 2.17+ on x86-64 or ARM64

## Usage

### Register a person

```console
$ facecheck register "Saeed Rohani"
```

A preview window shows a box around the face: amber while it waits ("Move
closer", "Look straight at the camera"), green while it takes samples. It
takes about a second once the face is steady. Names are not case-sensitive,
so "saeed rohani" is the same person.

The command prints the saved record as JSON:

```json
{
	"registered": true,
	"name": "Saeed Rohani",
	"registered_at": "2026-11-18T15:49:29Z",
	"samples": 5,
	"replaced": false,
	"similar_to": null,
	"measurement": { "face_shape": { "label": "oval", "...": "..." }, "...": "..." }
}
```

`measurement` holds the full face measurement described under Measurements
below. If the face already looks like someone who is
registered, `similar_to` names them and a warning is printed, which catches
the same person being registered twice under two names.

To register someone again under the same name, add `--replace`.

### Check who is in front of the camera

```console
$ facecheck check
```

```json
{
	"id": "7f864a9e6fa74e5fa2bf9eb4204e183f",
	"checked_at": "2026-11-18T15:49:29Z",
	"mode": "identify",
	"claimed_name": null,
	"matched": true,
	"name": "Saeed Rohani",
	"similarity": 0.9361,
	"threshold": 0.363,
	"candidates": [{ "name": "Saeed Rohani", "similarity": 0.9361 }],
	"frames_used": 3,
	"source": "camera:0"
}
```

The window shows the verdict in green ("Match: Saeed Rohani") or red ("No
match"). Every check is also logged in the database.

### Check one named person

```console
$ facecheck check --name "Saeed Rohani"
```

Same output, with `"mode": "verify"` and `claimed_name` filled in.

### Use it from a script

Only the JSON result goes to standard output; messages go to standard error.
The exit code tells you the outcome:

| Exit code | Meaning                                     |
| --------- | ------------------------------------------- |
| `0`       | Success; for `check`, a match               |
| `1`       | Error, cancelled, or no usable face in time |
| `2`       | Wrong command-line options                  |
| `3`       | `check` ran, but there was no match         |

```console
$ facecheck check --name "Saeed Rohani" --no-window > result.json && echo "Welcome"
```

### Other commands

| Command                 | What it does                                                                |
| ----------------------- | --------------------------------------------------------------------------- |
| `facecheck list`        | List registered people                                                      |
| `facecheck delete NAME` | Remove a person and all their face data                                     |
| `facecheck export`      | Write `users.csv`, `checks.csv` and `measurements.csv` here (or `--to DIR`) |
| `facecheck measure`     | Measure a face without registering anyone (see below)                       |

### Options

Options go after the command, for example `facecheck check --camera 1`.

| Option                   | Commands                 | Meaning                                                    |
| ------------------------ | ------------------------ | ---------------------------------------------------------- |
| `--camera N`             | register, check, measure | Camera index, default `0`. Try `1` for an external camera. |
| `--no-window`            | register, check, measure | No preview window.                                         |
| `--no-mirror`            | register, check, measure | Do not mirror the preview.                                 |
| `--timeout SEC`          | register, check, measure | Give up after this long, default `60`.                     |
| `--replace`              | register                 | Replace a person already registered under this name.       |
| `--samples N`            | register                 | Face samples to store, default `5`.                        |
| `--name NAME`            | check                    | Check against this person only.                            |
| `--threshold T`          | check                    | Similarity needed for a match, default `0.363`.            |
| `--ipd MM`, `--hfov DEG` | register, measure        | Scale settings for measurements (see below).               |
| `--data DIR`             | all                      | Folder for the database.                                   |
| `--model-dir DIR`        | all                      | Folder with your own copies of the model files.            |

### Run from the source folder, without installing

```console
$ pip install --only-binary=:all: opencv-contrib-python numpy
$ python run.py register "Saeed Rohani"
$ python run.py check
```

## Where the data is kept

The database is `facecheck.db` in:

| System  | Folder                                    |
| ------- | ----------------------------------------- |
| macOS   | `~/Library/Application Support/facecheck` |
| Windows | `%APPDATA%\facecheck`                     |
| Linux   | `~/.local/share/facecheck`                |

Use `--data DIR`, or set the `FACECHECK_HOME` environment variable, to keep
it somewhere else; for example, one folder per site or project.

Tables in the database:

| Table          | Contents                                                               |
| -------------- | ---------------------------------------------------------------------- |
| `users`        | name, registration time, number of samples, link to the measurement    |
| `face_samples` | each person's face prints (128 numbers each)                           |
| `measurements` | face measurements: one per registered person, plus `measure` results   |
| `checks`       | every check: when, which mode, who was claimed, who matched, the score |

`facecheck export` never writes face prints to CSV; they stay in the
database.

## How matching works

SFace turns a face into 128 numbers. Two photos of the same person give
nearly the same numbers, and the **similarity** (cosine similarity, from −1
to 1) measures how close they are. A person matches when the similarity
reaches the threshold, 0.363 by default. That is OpenCV's published
threshold for this model; on the LFW benchmark of face photos the model is
99.6% accurate with it.

Registration stores 5 samples taken a moment apart, and a person's score is
their best-matching sample. A check averages 3 consecutive frames of one
steady face.

- To turn away look-alikes more strictly, raise the threshold, for example
  `--threshold 0.5`. The right person will then be turned away more often,
  especially in poor light.
- If the right person is often not recognised, register them again in the
  light where the camera is used (`--replace`), or lower the threshold a
  little.

## Measurements

Each registration also saves the measurements and face-shape estimate of the
original `facemeasure` app. `facecheck measure` runs that app on its own:
`SPACE` saves, `E` exports CSV, `Q` quits; `--once` saves one result
automatically, and `--image photo.jpg` measures a photo.

It finds 12 lengths, 4 ratios, a jaw angle and the head pose, and estimates
the face shape: oval, round, square, oblong, heart, diamond or triangle.
Lengths use the standard 68-point landmark layout:

| Name                                                    | From → to                                             |
| ------------------------------------------------------- | ----------------------------------------------------- |
| `face_length`                                           | Estimated hairline → chin, using the "rule of thirds" |
| `face_width`                                            | Across the face at cheekbone level                    |
| `forehead_width`                                        | Outer ends of the eyebrows                            |
| `jaw_width`                                             | Across the jaw corners                                |
| `eye_distance_inner`, `eye_distance_outer`              | Inner and outer eye corners                           |
| `pupil_distance`                                        | Centres of the two eyes                               |
| `eye_width`, `nose_length`, `nose_width`, `mouth_width` | As named                                              |
| `lower_face_height`                                     | Base of the nose → chin                               |

Millimetres assume an average adult pupil distance of 63 mm. Pass the
person's real pupil distance (it is on most glasses prescriptions) with
`--ipd 61.5` for real-scale numbers. The face shape is a hand-tuned
heuristic, not a validated classifier.

## Limitations

Please read these before relying on `facecheck`.

- **It can be fooled by a photo.** There is no liveness or anti-spoofing
  check, so a printed photo or a phone screen showing a registered face may
  match. Do not use `facecheck` alone to protect anything valuable, such as
  doors, accounts or payments.
- **Matching is not perfect.** Look-alikes and twins can match each other,
  and poor light, strong angles, masks or heavy changes in appearance can
  stop the right person matching. The threshold trades one error for the
  other.
- **It has been tested with synthetic data only.** The automated tests use a
  synthetic face and fake camera and models, and the real models have not
  yet been tried together on a large set of real faces. Try it in your own
  setting before you depend on it.
- **Measurements are estimates.** See above.

## Privacy and consent

Face prints are biometric data, which many laws treat as sensitive
personal data (for example the GDPR in the EU and BIPA in Illinois). If you
register anyone other than yourself:

- Tell them what is stored and why, and get their consent first.
- Keep the database private, and delete people (`facecheck delete NAME`)
  when their data is no longer needed.
- Never commit `facecheck.db` or exported CSV files to a public repository.

Frames are processed in memory and discarded; no image is written to disk.
The only network requests are the one-time model downloads.

## Use from Python

```python
import cv2
from facecheck import FaceDetector, Store, decide, face_problem, mean_embedding

with FaceDetector(landmarks=False, recognition=True) as detector, Store("people.db") as store:
    frame = cv2.imread("visitor.jpg")
    face = detector.find_face(frame)
    if face_problem(face) is None:
        result = decide(mean_embedding([detector.embedding(frame, face)]), store.gallery())
        print(result["matched"], result["name"], result["similarity"])
```

`store.add_user(name, embeddings, measurement=None)` registers someone the
same way.

## Troubleshooting

**pip is stuck at `Building wheel for opencv-contrib-python`.** pip found no
ready-made OpenCV for your system and is compiling it. Stop it with `Ctrl+C`
and install with `--only-binary`, as shown under Install. If pip then finds
no OpenCV at all, check your system against the requirements:

```console
$ python -c "import sys, platform; print(sys.version); print(platform.platform())"
```

**The download fails with `CERTIFICATE_VERIFY_FAILED`.** Your Python has no
list of trusted HTTPS certificates; this is common with the python.org
installer on macOS. `facecheck` then retries with the system's `curl`. To fix
Python itself, open the Python folder in Applications and run
`Install Certificates.command`.

**A model download fails for another reason.** Download the file yourself
into the cache folder that the error message names (on macOS
`~/Library/Caches/facecheck`) and run again:

```console
$ mkdir -p ~/Library/Caches/facecheck && cd ~/Library/Caches/facecheck
$ curl -L -O https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
$ curl -L -O https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
$ curl -L -O https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
```

**"OpenCV's face module is missing".** Another OpenCV build is hiding the
contrib one. Remove them all and reinstall only the contrib build:

```console
$ pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
$ pip install --only-binary=:all: opencv-contrib-python
```

**The camera does not open.** Close other apps that use it and try
`--camera 1`. On macOS, allow your terminal under System Settings → Privacy &
Security → Camera, then restart the terminal.

**The box stays amber.** Follow the message: move closer, look straight at
the camera, and add light from the front.

## Development

```console
$ git clone https://github.com/YOUR-GITHUB-USERNAME/facecheck.git
$ cd facecheck
$ python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
$ pip install --only-binary=opencv-contrib-python,numpy -e ".[test]"
$ pytest
```

The tests need neither a camera nor the model files. Release steps are in
`PUBLISHING.md`.

## Credits

- [OpenCV](https://opencv.org/), and the YuNet and SFace models from the
  [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- Facemark LBF and its trained model by Laksono Kurnianggoro
  ([GSoC 2017](https://github.com/kurnianggoro/GSOC2017))
- The generic 3D face used for measurements comes from MediaPipe's canonical
  face model, by Google (Apache License 2.0)
- [NumPy](https://numpy.org/)

The model files are not part of this package; they are downloaded from their
authors' repositories. Check each model's terms before commercial use: the
OpenCV Zoo keeps a licence file next to each model, and the LBF model's
repository states no licence.

## Author

Created and maintained by **Saeed Rohani**.

## License

MIT © 2026 Saeed Rohani. See the `LICENSE` file.
