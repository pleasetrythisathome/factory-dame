(ns neurodynamics.synths
  "Synth definitions for the ground-truth harness.

  Everything a modular patch gives us, expressed as code: oscillator
  choice, filter type + envelope modulation, amp envelope. No cables,
  no knobs, no routing surprises. Change a parameter, save, re-render
  — the audio output is fully determined by what's written here.

  Conventions:
  - ``freq`` in Hz (the trigger pattern's ``freq_hz`` passes straight in).
  - ``gate`` drives an ADSR; pass ``0`` to trigger the release and free
    the synth.
  - ``amp`` pre-envelope gain, 0..1.
  - Default envelope + filter settings are tuned for the kinds of
    broadband, transient-rich signals the NRT engine expects — roughly
    equivalent to a saw-into-resonant-lowpass Eurorack patch with
    envelope on cutoff."
  (:require [overtone.core :as o]))


(o/defsynth mono-saw
  "One-oscillator saw through a resonant low-pass filter with an
   envelope-modulated cutoff plus an amp envelope. Most-generic
   subtractive monosynth — the ``default modular patch``."
  [freq     440
   gate     1
   amp      0.5
   ;; Amp envelope (ADSR)
   atk      0.005
   dec      0.08
   sus      0.7
   rel      0.15
   ;; Filter envelope (cutoff sweep on each note)
   f-base   400
   f-peak   3500
   f-atk    0.003
   f-dec    0.08
   f-sus    0.4
   f-rel    0.2
   res      0.35
   ;; Stereo pan for voice tests
   pan      0]
  (let [amp-env  (o/env-gen (o/adsr atk dec sus rel) gate :action o/FREE)
        f-env    (o/env-gen (o/adsr f-atk f-dec f-sus f-rel) gate)
        cutoff   (+ f-base (* f-env (- f-peak f-base)))
        osc      (o/saw freq)
        filt     (o/rlpf osc cutoff res)
        stereo   (o/pan2 (* amp amp-env filt) pan)]
    (o/out 0 stereo)))


(o/defsynth plucked
  "Short percussive note — faster envelopes, lower sustain. Useful for
   hi-hat/stab patterns where a sustain would smear voice extraction."
  [freq 440 gate 1 amp 0.5
   atk 0.002 dec 0.04 sus 0.0 rel 0.08
   f-base 800 f-peak 5000 res 0.2 pan 0]
  (let [amp-env (o/env-gen (o/adsr atk dec sus rel) gate :action o/FREE)
        f-env   (o/env-gen (o/adsr 0.001 0.05 0 0.1) gate)
        cutoff  (+ f-base (* f-env (- f-peak f-base)))
        osc     (o/saw freq)
        filt    (o/rlpf osc cutoff res)]
    (o/out 0 (o/pan2 (* amp amp-env filt) pan))))


(o/defsynth sine-tone
  "Pure sine at a given freq — the simplest reference. Use this when
   debugging to check the capture path preserves exactly one pitch."
  [freq 440 gate 1 amp 0.5
   atk 0.005 dec 0.05 sus 0.9 rel 0.1 pan 0]
  (let [env (o/env-gen (o/adsr atk dec sus rel) gate :action o/FREE)]
    (o/out 0 (o/pan2 (* amp env (o/sin-osc freq)) pan))))


(def synth-registry
  "Name → synth-def lookup so the CLI can pick by string."
  {"mono-saw"   mono-saw
   "plucked"    plucked
   "sine-tone"  sine-tone})


(defn pick-synth
  "Return the synth-def for `name`, or throw with available options."
  [name]
  (or (get synth-registry name)
      (throw (ex-info (str "unknown synth '" name "'")
                      {:available (keys synth-registry)}))))
