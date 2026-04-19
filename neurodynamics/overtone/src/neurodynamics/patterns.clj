(ns neurodynamics.patterns
  "Load trigger patterns from the engine's JSON schedule and play them
  through an Overtone synth.

  The Python trigger-roundtrip harness writes each pattern to
  ``neurodynamics/engine/test_audio/triggers/<name>.triggers.json`` —
  this namespace reads that same file so ground truth stays
  identical across the Python and Clojure paths.

  Schedule semantics: every trigger is ``{:start_s, :duration_s,
  :freq_hz, :velocity}``. ``play-pattern`` schedules each trigger
  relative to a single start-time anchor so timing stays consistent
  even as the JVM runs other work."
  (:require [cheshire.core :as json]
            [clojure.java.io :as io]
            [overtone.core :as o]
            [neurodynamics.synths :as s]))


(def ^:dynamic *engine-root*
  "Resolved at load time from project layout: the Python engine's
  trigger directory. If run from outside the repo, override via the
  ``ND_ENGINE_TRIGGERS`` env var."
  (or (System/getenv "ND_ENGINE_TRIGGERS")
      (let [here (io/file (System/getProperty "user.dir"))]
        (.getCanonicalPath
          (io/file here ".." "engine" "test_audio" "triggers")))))


(defn load-pattern
  "Read ``<engine>/test_audio/triggers/<name>.triggers.json`` and
  return {:name, :duration_s, :triggers [{:start_s :duration_s
  :freq_hz :velocity}]}."
  [pattern-name]
  (let [path (io/file *engine-root* (str pattern-name ".triggers.json"))]
    (when-not (.exists path)
      (throw (ex-info (str "no pattern file at " path
                           " — run tests.trigger_roundtrip with any pattern "
                           "to generate it first")
                      {:pattern pattern-name :path (str path)})))
    (with-open [r (io/reader path)]
      (let [raw (json/parse-stream r true)]
        (-> raw
            (update :triggers
                    (fn [ts] (mapv #(update % :velocity (fnil identity 1.0)) ts))))))))


(defn play-pattern
  "Schedule every trigger in `pattern` on `synth-def`. Returns the
  absolute scheduler time (ms since epoch) at which the last note
  releases, so the caller can wait that long before teardown."
  [synth-def pattern & {:keys [start-offset-ms]
                         :or {start-offset-ms 500}}]
  (let [t0 (+ (o/now) start-offset-ms)
        last-end (atom t0)]
    (doseq [trig (:triggers pattern)]
      (let [start-ms (+ t0 (int (* 1000 (:start_s trig))))
            end-ms   (+ start-ms (int (* 1000 (:duration_s trig))))]
        (when (> end-ms @last-end) (reset! last-end end-ms))
        (o/at start-ms
          (let [s (synth-def :freq (double (:freq_hz trig))
                              :amp (* 0.5 (double (:velocity trig)))
                              :gate 1)]
            (o/at end-ms
              (o/ctl s :gate 0))))))
    @last-end))


(defn total-duration-ms
  "How long to wait (from now) before every scheduled note has
  finished releasing. Includes a generous tail for envelope release."
  [pattern & {:keys [start-offset-ms tail-ms]
              :or {start-offset-ms 500 tail-ms 800}}]
  (let [trigger-end (apply max 0
                            (map #(int (* 1000 (+ (:start_s %)
                                                   (:duration_s %))))
                                 (:triggers pattern)))]
    (+ start-offset-ms trigger-end tail-ms)))
