# Synthetic yoga & health dataset v2 (SIMULATED)

Built for prototyping pipelines. Effects are assumptions written into the generator,
not findings; models that recover them only recover those assumptions.

- subjects.csv: 200 subjects (demographics, condition group, baseline HR/HRV/stress/sleep/pain/mood/energy, practice profile)
- sessions.csv: 6361 sessions with pre/post HR, HRV (RMSSD), respiration, stress VAS, pose_quality, balance_score
- weekly_outcomes.csv: 2400 rows, weeks 1-12: stress, sleep_hrs, mood, energy, pain, sessions_that_week
- pose_keypoints.json: 600 pose records (2 poses x 300 sessions), 33 landmarks x [x, y, z, visibility]

Keys: subject_id, session_id. Built-in structure: cumulative practice dose and calming
styles (pranayama, restorative, yin) increase HRV gains and stress reduction; outcome
effects depend on condition group and follow a saturating 12-week ramp.
Split by subject_id when modelling.
