# Odolite demo: voiceover script

Video: `odolite_demo.mp4`, 1920×1080, 4:52, silent track (record the voice and lay it over, or replace the
audio track). Made by `record_demo.py` from the real UI. Every number on screen comes from the committed
case and evaluation files, and the upload scene is a live analysis by the local server.

Pace: about 140 words a minute, calm. Each block starts at its timecode. If a block runs short, let the
picture breathe; if it runs long, cut the sentence marked *(optional)*.

Rules check (opencv.org/opencv-ai-competition-2026): at most 5 minutes ✓ (4:52); it must show the team
(0:00), the application working (0:12–3:36, including a live upload at 3:10), its architecture (3:36) and
its principal results (4:04). Upload it as public or unlisted (YouTube unlisted is fine) and check that the
link opens while logged out.

---

### 0:00–0:12 · Title card

> Hi, I'm Abdullah. This is Odolite, my entry for the OpenCV AI Competition. It turns ordinary road footage
> into physical measurements, and states every assumption they rest on.

### 0:12–1:00 · Real dashcam, the vehicle ahead

*(case opens and plays; at about 0:20 it scrolls to the distance charts; at about 0:47 to the time-to-collision table)*

> Real footage first: one minute of driving from the comma2k19 dataset.
>
> This is the car ahead of us. A YOLOX detector, running on OpenCV 5's DNN module, finds it in our lane. Its
> distance comes from two independent rulers: where its tyres meet the road, and how wide it looks. If the
> two disagree, Odolite refuses that window instead of guessing.
>
> The grey line is the car's own radar, which Odolite never saw. The test was pre-registered: the first
> twenty seconds for development, the last forty run once. Distance is within 2.4 percent, and every interval
> contains the radar. Closing speed is within 0.6 metres per second.

### 1:00–1:21 · Same clip, own speed: a refusal

> Same clip, our own speed. Under a freeway marking standard, the lane width and the camera's focal length
> contradict that assumption, so Odolite answers none of the windows, instead of reporting speeds twice too
> high. This was our pre-registered holdout. Its first version failed, and we publish that.

### 1:21–1:50 · Fixed roadside camera

*(the clip plays twice, the second time slowed down)*

> Now a fixed roadside camera. Here the only ruler is the paint. OpenCV finds the lane lines and their
> vanishing point. Because the three lines are equally spaced, the horizon, and the camera's roll, follow in
> closed form. The grid is the recovered road plane, drawn back onto the video. MOG2, a Kalman filter and
> Hungarian assignment track the vehicles.

### 1:50–2:29 · Which assumption decides the answer

*(envelope map, then "How close they came", then "Checked against ground truth")*

> Could braking have prevented this? Odolite doesn't pick one reaction time. It replays the approach for
> every reaction from 0.75 to two seconds and every braking level from five to eight metres per second
> squared. It only says "avoidable" or "not avoidable" if the answer holds inside the whole box. Here it
> doesn't, so it says which assumption decides it.
>
> These are the standard near-miss measures: smallest gap, time to collision and braking needed. This clip
> is synthetic, so we can check against ground truth, including the two rows where Odolite is off.

### 2:29–3:10 · The investigator

*(camera-knocked case; scrolls through "What Odolite tried after the dispute")*

> Every result is cross-examined with evidence the fit never used. In this clip the camera was knocked
> halfway through, and the check catches it.
>
> Odolite then runs the remedy for that specific failure. It finds the bump with OpenCV's ECC image
> alignment, splits the clip, and analyses each part again. Each part has to pass the whole
> cross-examination from scratch. Neither does, so the refusal stands, and Odolite says what input would
> help. It never relaxes a threshold to produce an answer.

### 3:10–3:36 · Live upload

*(file chosen, frame rate typed, Analyse; the wait is cut; the new case appears)*

> This is the app working live. I upload a clip and tell it the frame rate, which is the one thing the file
> metadata often gets wrong. The analysis takes about half a minute, cut here. The result comes back with
> the same reasoning, cross-examination and stated assumptions.

### 3:36–4:04 · Architecture

> OpenCV 5 does every image operation: the markings, tracking, optical flow, the image alignment and the DNN
> detector. On AWS, the browser uploads through a presigned URL to a private S3 bucket. The upload triggers a
> container Lambda that runs the full analysis, DynamoDB tracks the job, and footage expires automatically
> after seven days.

### 4:04–4:42 · Results, failures included

*(test3 table, then "What the investigator changed", then the radar table)*

> Results, failures included. Each test set was fixed in advance and run once. With true tracks, the
> reasoning made no harmful errors on the newest held-out split. From raw pixels it got five of thirty right
> and made five harmful errors, mostly when the following car was hidden. That's the next thing to fix.
> The investigator changed no outcome there. Collision verdicts are validated on simulation only, because we
> found no real crash footage we could legally use. The dashcam measurements are validated on real footage.

### 4:42–4:52 · Closing card

> Odolite never assigns fault. It measures what the footage supports, and says why when it can't. Thanks for
> watching.

---

## Recording the voice

* Phone voice memo in a quiet room is fine; keep the phone about 20 cm away.
* Lay it over the video: in CapCut, DaVinci Resolve or iMovie, drop the video and the voice on the timeline
  and nudge each block to its timecode. Or with ffmpeg:
  `ffmpeg -i odolite_demo.mp4 -i voice.m4a -map 0:v -map 1:a -c:v copy -c:a aac -shortest odolite_final.mp4`
* Optional: add captions (YouTube auto-captions, then correct them), which judges watching on mute appreciate.
