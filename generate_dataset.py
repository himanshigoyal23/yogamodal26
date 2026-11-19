"""Synthetic yoga & health dataset v2 (SIMULATED - for pipeline prototyping only).

subjects.csv          200 subjects: demographics, baseline health, practice profile
sessions.csv          per-session pre/post HR, HRV (RMSSD), respiration, stress, pose/balance scores
weekly_outcomes.csv   12-week trajectories: stress, sleep, mood, energy, pain
pose_keypoints.json   MediaPipe-style 33-landmark keypoints for a sample of sessions
Run: python generate_dataset.py
"""
import json, os
import numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
N, WEEKS = 200, 12
STYLES = ["hatha", "vinyasa", "pranayama", "restorative", "yin"]
CALM = {"pranayama": 1.0, "restorative": 1.0, "yin": .8, "hatha": .5, "vinyasa": .1}
COND = ["healthy", "hypertension", "anxiety", "low_back_pain", "poor_sleep"]

# ---------------------------------------------------------------- subjects
cond = rng.choice(COND, N)
sub = pd.DataFrame(dict(
    subject_id=[f"S{i:03d}" for i in range(N)],
    age=rng.integers(20, 70, N),
    sex=rng.choice(["F", "M"], N, p=[.55, .45]),
    bmi=rng.normal(25.5, 4, N).clip(17, 41).round(1),
    condition_group=cond,
    experience_years=rng.gamma(1.5, 1.5, N).clip(0, 15).round(1),
    practice_freq_per_week=rng.choice([1, 2, 3, 4, 5, 6], N, p=[.1, .2, .3, .2, .13, .07]),
    primary_style=rng.choice(STYLES, N, p=[.3, .25, .15, .15, .15]),
))
a = sub.age.values
sub["resting_hr"] = (66 + 5 * (cond == "hypertension") + 4 * (cond == "anxiety")
                     + .05 * (sub.bmi - 25) * 3 + rng.normal(0, 6, N)).clip(45, 100).round(0)
sub["baseline_hrv_rmssd"] = (55 - .45 * (a - 20) - 5 * (cond == "anxiety")
                             + rng.normal(0, 10, N)).clip(10, 100).round(1)
sub["baseline_stress"] = (14 + 10 * (cond == "anxiety") + rng.normal(0, 4, N)).clip(0, 40).round(0)
sub["baseline_sleep_hrs"] = (7 - 1.2 * (cond == "poor_sleep") - .5 * (cond == "anxiety")
                             + rng.normal(0, .6, N)).clip(3.5, 9.5).round(1)
sub["baseline_pain"] = (1.5 + 4 * (cond == "low_back_pain") + rng.normal(0, 1.2, N)).clip(0, 10).round(1)
sub["baseline_mood"] = (6 - 2 * (cond == "anxiety") + rng.normal(0, 1.2, N)).clip(1, 10).round(1)
sub["baseline_energy"] = (5.5 - 1.2 * (cond == "poor_sleep") + rng.normal(0, 1.3, N)).clip(1, 10).round(1)
sub.to_csv(f"{OUT}/subjects.csv", index=False)

# ---------------------------------------------------------------- sessions
rows, sid = [], 0
for i, s in sub.iterrows():
    done = 0
    for wk in range(1, WEEKS + 1):
        for _ in range(int(s.practice_freq_per_week)):
            if rng.random() > .8:
                continue
            sid += 1; done += 1
            st = s.primary_style if rng.random() < .7 else rng.choice(STYLES)
            calm = CALM[st]
            dose = min(done / 30, 1.0)                    # cumulative practice
            pre_hr = s.resting_hr + rng.normal(0, 3)
            pre_hrv = s.baseline_hrv_rmssd * rng.normal(1, .06)
            pre_resp = 15 + rng.normal(0, 1.2)
            pre_str = np.clip(3 + s.baseline_stress / 8 + rng.normal(0, 1), 0, 10)
            post_hr = pre_hr - (2 + 5 * calm) * (1 + .5 * dose) + rng.normal(0, 2)
            post_hrv = pre_hrv * (1 + (.06 + .14 * calm) * (1 + dose)) + rng.normal(0, 2)
            post_resp = pre_resp - (1 + 3 * calm) + rng.normal(0, .8)
            post_str = np.clip(pre_str - (1 + 2 * calm) * (1 + .3 * dose) + rng.normal(0, .8), 0, 10)
            skill = 1 - np.exp(-(s.experience_years + done / 20) / 3)
            rows.append(dict(session_id=f"X{sid:05d}", subject_id=s.subject_id, week=wk,
                style=st, duration_min=int(rng.choice([30, 45, 60, 75])),
                pre_hr=round(pre_hr, 1), post_hr=round(post_hr, 1),
                pre_hrv_rmssd=round(pre_hrv, 1), post_hrv_rmssd=round(post_hrv, 1),
                pre_resp_rpm=round(pre_resp, 1), post_resp_rpm=round(post_resp, 1),
                pre_stress_vas=round(pre_str, 1), post_stress_vas=round(post_str, 1),
                pose_quality=round(float(np.clip(.35 + .5 * skill + rng.normal(0, .07), 0, 1)), 3),
                balance_score=round(float(np.clip(.3 + .55 * skill + rng.normal(0, .09), 0, 1)), 3)))
ses = pd.DataFrame(rows)
ses.to_csv(f"{OUT}/sessions.csv", index=False)

# ---------------------------------------------------------- weekly outcomes
wk_n = ses.groupby(["subject_id", "week"]).size().unstack(fill_value=0).reindex(
    columns=range(1, WEEKS + 1), fill_value=0)
wrows = []
for i, s in sub.iterrows():
    cum = 0
    for wk in range(1, WEEKS + 1):
        cum += wk_n.loc[s.subject_id, wk]
        dose = min(cum / 30, 1.0)
        ramp = 1 - np.exp(-wk / 4)
        e = dose * ramp
        c = s.condition_group
        wrows.append(dict(subject_id=s.subject_id, week=wk, sessions_that_week=int(wk_n.loc[s.subject_id, wk]),
            stress=round(float(np.clip(s.baseline_stress - (4 + 6 * (c == "anxiety")) * e + rng.normal(0, 2), 0, 40)), 1),
            sleep_hrs=round(float(np.clip(s.baseline_sleep_hrs + (.4 + .8 * (c == "poor_sleep")) * e + rng.normal(0, .35), 3, 10)), 1),
            mood=round(float(np.clip(s.baseline_mood + (1 + 1 * (c == "anxiety")) * e + rng.normal(0, .6), 1, 10)), 1),
            energy=round(float(np.clip(s.baseline_energy + 1.2 * e + rng.normal(0, .6), 1, 10)), 1),
            pain=round(float(np.clip(s.baseline_pain - (.6 + 2.2 * (c == "low_back_pain")) * e + rng.normal(0, .6), 0, 10)), 1)))
pd.DataFrame(wrows).to_csv(f"{OUT}/weekly_outcomes.csv", index=False)

# ------------------------------------------------- pose keypoints (33 landmarks)
LM = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
      "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
      "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
      "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
      "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
      "right_heel", "left_foot_index", "right_foot_index"]
# angles (deg; 0 = up, 90 = right): torso neck ual uar fal far thl thr shl shr
POSES = {"mountain": [0, 0, 180, 180, 180, 180, 180, 180, 180, 180],
         "tree": [0, 0, 10, 350, 0, 0, 180, 110, 180, 225],
         "warrior_ii": [0, 0, 270, 90, 270, 90, 250, 110, 250, 180],
         "downward_dog": [135, 150, 160, 160, 160, 160, 200, 200, 200, 200],
         "childs_pose": [100, 110, 95, 95, 90, 90, 90, 90, 270, 270]}
d = lambda t: np.array([np.sin(np.deg2rad(t)), np.cos(np.deg2rad(t))])

def landmarks(ang, rs):
    to, ne, ual, uar, fal, far, thl, thr, shl, shr = ang
    hl, hr_ = np.array([-.07, 0.]), np.array([.07, 0.])
    nb = .5 * d(to); sl, sr = nb + [-.1, 0], nb + [.1, 0]
    nose = nb + .25 * d(ne)
    el, er = sl + .3 * d(ual), sr + .3 * d(uar)
    wl, wr = el + .25 * d(fal), er + .25 * d(far)
    kl, kr = hl + .45 * d(thl), hr_ + .45 * d(thr)
    al, ar_ = kl + .45 * d(shl), kr + .45 * d(shr)
    hand = lambda w, f, sgn: (w + .07 * d(f) + sgn * .012 * d(f + 90),
                              w + .08 * d(f), w + .04 * d(f) - sgn * .012 * d(f + 90))
    pl, il, tl = hand(wl, fal, 1); pr, ir, tr = hand(wr, far, -1)
    P = {"nose": nose, "left_eye_inner": nose + [-.012, .02], "left_eye": nose + [-.025, .022],
         "left_eye_outer": nose + [-.04, .02], "right_eye_inner": nose + [.012, .02],
         "right_eye": nose + [.025, .022], "right_eye_outer": nose + [.04, .02],
         "left_ear": nose + [-.06, .0], "right_ear": nose + [.06, .0],
         "mouth_left": nose + [-.015, -.04], "mouth_right": nose + [.015, -.04],
         "left_shoulder": sl, "right_shoulder": sr, "left_elbow": el, "right_elbow": er,
         "left_wrist": wl, "right_wrist": wr, "left_pinky": pl, "right_pinky": pr,
         "left_index": il, "right_index": ir, "left_thumb": tl, "right_thumb": tr,
         "left_hip": hl, "right_hip": hr_, "left_knee": kl, "right_knee": kr,
         "left_ankle": al, "right_ankle": ar_, "left_heel": al + [-.02, -.035],
         "right_heel": ar_ + [.02, -.035], "left_foot_index": al + [-.07, -.04],
         "right_foot_index": ar_ + [.07, -.04]}
    arr = np.stack([P[n] for n in LM]) * rs
    x, y = .5 + arr[:, 0] * .45, .55 - arr[:, 1] * .45          # image coords, y down
    z = rng.normal(0, .03, len(LM))
    v = np.clip(rng.normal(.96, .03, len(LM)), .5, 1)
    return np.stack([x, y, z, v], 1).round(4).tolist()

sample = ses.sample(300, random_state=1)
exp_map = sub.set_index("subject_id").experience_years
recs = []
for _, r in sample.iterrows():
    sd = max(3, 10 - 6 * r.pose_quality)
    for pn in rng.choice(list(POSES), 2, replace=False):
        ang = np.array(POSES[pn], float) + rng.normal(0, sd, 10)
        recs.append(dict(session_id=r.session_id, subject_id=r.subject_id, pose=str(pn),
                         pose_quality=r.pose_quality, landmarks=landmarks(ang, rng.normal(1, .05))))
json.dump(dict(format="MediaPipe-style 33 landmarks; each row [x, y, z, visibility], x/y normalized image coords",
               landmark_names=LM, records=recs), open(f"{OUT}/pose_keypoints.json", "w"))

open(f"{OUT}/README.md", "w").write(f"""# Synthetic yoga & health dataset v2 (SIMULATED)

Built for prototyping pipelines. Effects are assumptions written into the generator,
not findings; models that recover them only recover those assumptions.

- subjects.csv: {N} subjects (demographics, condition group, baseline HR/HRV/stress/sleep/pain/mood/energy, practice profile)
- sessions.csv: {len(ses)} sessions with pre/post HR, HRV (RMSSD), respiration, stress VAS, pose_quality, balance_score
- weekly_outcomes.csv: {N * WEEKS} rows, weeks 1-{WEEKS}: stress, sleep_hrs, mood, energy, pain, sessions_that_week
- pose_keypoints.json: {len(recs)} pose records (2 poses x 300 sessions), 33 landmarks x [x, y, z, visibility]

Keys: subject_id, session_id. Built-in structure: cumulative practice dose and calming
styles (pranayama, restorative, yin) increase HRV gains and stress reduction; outcome
effects depend on condition group and follow a saturating 12-week ramp.
Split by subject_id when modelling.
""")
print(len(sub), len(ses), len(wrows), len(recs))
