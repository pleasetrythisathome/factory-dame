(ns neurodynamics.render
  "Offline render: play a trigger pattern through a synth, record the
  SC server output to a WAV, stop. Uses Overtone's live-server
  recording (so it runs in real time, not true non-realtime — but
  produces a deterministic WAV either way for the voice-extraction
  compare).

  Usage (from neurodynamics/overtone):
      clojure -M:render --pattern single --synth mono-saw
      clojure -M:render --pattern bassline --synth plucked
      clojure -M:render --pattern all

  Output goes to ``<engine>/test_audio/triggers/<name>.overtone.wav``
  so the trigger-roundtrip harness can compare it alongside
  ``<name>.vcv.wav`` (same directory, different synthesis source)."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [overtone.core :as o]
            [neurodynamics.synths :as s]
            [neurodynamics.patterns :as p])
  (:gen-class))


(defn- engine-root []
  (.getCanonicalPath (io/file p/*engine-root*)))


(defn- wait [ms]
  (Thread/sleep (max 0 (int ms))))


(defn render-one
  "Render `pattern-name` through `synth-name` to a WAV at
  ``<engine>/test_audio/triggers/<pattern>.overtone.wav``. Returns the
  absolute output path."
  [pattern-name synth-name]
  (let [pattern (p/load-pattern pattern-name)
        synth   (s/pick-synth synth-name)
        out-dir (io/file (engine-root))
        _       (.mkdirs out-dir)
        out     (io/file out-dir (str pattern-name ".overtone.wav"))
        total   (p/total-duration-ms pattern)]
    (println (format "  rendering %s via %s → %s (%d ms)"
                     pattern-name synth-name (.getName out) total))
    ;; recording-start writes the SC server output bus directly to a
    ;; 32-bit float WAV. Record a bit before the first scheduled note
    ;; so attack transients aren't clipped.
    (o/recording-start (.getCanonicalPath out))
    (p/play-pattern synth pattern)
    (wait total)
    (o/recording-stop)
    ;; Give SC a moment to flush the WAV header.
    (wait 200)
    (.getCanonicalPath out)))


(defn- parse-args [argv]
  (let [args (into {} (partition-all 2) argv)]
    {:pattern (or (get args "--pattern") "single")
     :synth   (or (get args "--synth") "mono-saw")}))


(defn -main [& argv]
  (let [{:keys [pattern synth]} (parse-args argv)
        patterns (if (= pattern "all")
                   ["single" "quarter_notes" "bassline"
                    "chord_progression" "polyrhythm"]
                   [pattern])]
    (println "Booting SuperCollider server…")
    (o/boot-server)
    (try
      (doseq [p patterns]
        (try
          (let [out (render-one p synth)]
            (println (str "  → " out)))
          (catch Exception e
            (println (str "  [error] " (.getMessage e))))))
      (finally
        (o/kill-server)))
    (System/exit 0)))
