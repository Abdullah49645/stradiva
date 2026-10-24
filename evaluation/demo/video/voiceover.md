# Stradiva demo: voiceover script

Video: `stradiva_demo.mp4`, 1920×1080, 4:37. The audio track is silent: record your voice and lay it over.
`record_demo.py` made the video from the real UI. Every number on screen comes from the committed case
and evaluation files, and the upload scene is a live analysis by the local server.

Pace: about 140 words a minute, calm. Each block starts at its timecode. If a block runs short, let the
picture breathe. If it runs long, drop the sentence marked *(optional)*.

Rules check (opencv.org/opencv-ai-competition-2026):
* at most 5 minutes: ✓ (4:37);
* shows the team: 0:00;
* shows the application working: 0:13–3:21, including a live upload at 2:08;
* shows its architecture: 3:21;
* shows its principal results: 3:49.

Upload it public or unlisted (unlisted YouTube is fine), and check that the link opens while logged out.

---

### 0:00–0:13 · Title card

> Hi, I'm Abdullah. This is Stradiva. Give it collision footage from an ordinary, uncalibrated camera, and it
> works out the physics, stating every assumption along the way.

### 0:13–0:48 · A collision on a fixed roadside camera

*(the clip plays twice, the second time slowed down; then the key facts)*

> A rear-end collision on a roadside camera. It's from our simulator, so every number can be checked against
> ground truth. The only ruler is the road paint. OpenCV finds the lane lines and their vanishing point, and
> because they're equally spaced, the horizon and camera roll follow in closed form. The grid is the recovered road plane, drawn back onto the video. MOG2, a Kalman
> filter and Hungarian assignment track the vehicles. Out come each vehicle's speed, when braking began,
> and the moment of contact.

### 0:48–1:26 · Could braking have prevented it?

*(envelope map, then "How close they came", then "Checked against ground truth")*

> Could braking have prevented it? Stradiva doesn't pick one reaction time. It replays the approach for every
> reaction from 0.75 to two seconds and every braking level from five to eight metres per second squared.
> It only says "avoidable" or "not avoidable" if the answer holds inside the whole box. Here it doesn't, so
> it says which assumption decides it.
>
> Below are the standard near-miss measures: smallest gap, time to collision and braking needed. Then the
> check against ground truth, including the two rows where Stradiva is off.

### 1:26–2:08 · When the footage fights back

*(camera-knocked case; scrolls through "What Stradiva tried after the dispute")*

> Every result is cross-examined with evidence the fit never used. In this clip the camera was knocked
> halfway through, and the check catches it.
>
> Stradiva then runs the remedy for that specific failure. It finds the bump with OpenCV's ECC image
> alignment, splits the clip, and analyses each part again. Each part has to pass the whole
> cross-examination from scratch. Neither does, so the refusal stands, and Stradiva says what input would
> help. It never relaxes a threshold to produce an answer.

### 2:08–2:33 · Live upload

*(file chosen, frame rate typed, Analyse; the wait is cut; the new case appears)*

> This is the app working live. I upload a collision clip and give it the frame rate, the one thing file
> metadata often gets wrong. The analysis takes about half a minute, cut here. Back comes the full case:
> speeds, contact, the cross-examination and every stated assumption.

### 2:33–3:21 · Added mode: dashcam footage

*(real dashcam clip, the vehicle-ahead charts, then the own-speed refusal)*

> Stradiva also handles a moving camera, as an added mode. This is real footage: one minute of driving from
> the comma2k19 dataset. A YOLOX detector on OpenCV 5's DNN module finds the car ahead. Its distance comes
> from two independent rulers: where its tyres meet the road, and how wide it looks. If they disagree, the
> window is refused. The grey line is the car's own radar, which Stradiva never saw. On the pre-registered
> test, distance was within 2.4 percent, and every interval contained the radar.
>
> On the same clip, the lane width contradicts a freeway marking assumption, so Stradiva refuses to give
> speeds that would be twice too high.

### 3:21–3:49 · Architecture

> OpenCV 5 does every image operation: the markings, tracking, optical flow, the image alignment and the DNN
> detector. On AWS, the browser uploads through a presigned URL to a private S3 bucket. The upload triggers a
> container Lambda that runs the full analysis, DynamoDB tracks the job, and footage expires automatically
> after seven days.

### 3:49–4:27 · Results, failures included

*(test3 table, then "What the investigator changed", then the radar table)*

> Results, failures included. Each test set was fixed in advance and run once. With true tracks, the
> collision reasoning made no harmful errors on the newest held-out split. From raw pixels, it got five of
> thirty right and made five harmful errors, mostly when the following car was hidden. That's the next
> thing to fix. Collision verdicts are validated on simulation only, because we found no real crash footage
> we could legally use. The dashcam measurements are validated on real footage.

### 4:27–4:37 · Closing card

> Stradiva never assigns fault. It measures what the footage supports, and says why when it can't. Thanks for
> watching.

---

## Recording the voice

* A phone voice memo in a quiet room is fine; hold the phone about 20 cm away.
* To lay it over the video, put the video and the voice on one timeline in CapCut, DaVinci Resolve or
  iMovie, and nudge each block to its timecode. Or, with ffmpeg:
  `ffmpeg -i stradiva_demo.mp4 -i voice.m4a -map 0:v -map 1:a -c:v copy -c:a aac -shortest stradiva_final.mp4`
* Optional: add captions for judges watching on mute (YouTube auto-captions, then correct them).
