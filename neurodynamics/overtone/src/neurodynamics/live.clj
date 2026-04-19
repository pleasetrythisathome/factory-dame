(ns neurodynamics.live
  "Live playback: play a trigger pattern through a synth directly to
  the SuperCollider server's audio output. If SC is configured to
  use the Loopback Audio device, ``nd-live`` can read that output
  and extract voices in real time.

  Usage (from neurodynamics/overtone):
      clojure -M:live --pattern single --synth mono-saw
      clojure -M:live --pattern bassline

  Set the SC output device to Loopback Audio in one of two ways:
  1. In system Audio MIDI Setup, set Loopback Audio as the default
     output, then boot SC before running anything else.
  2. Pass an explicit device name via the ``SC_HW_DEVICE_NAME`` env
     var (e.g. ``SC_HW_DEVICE_NAME=\"Loopback Audio\" clojure -M:live …``);
     Overtone forwards this to scsynth's ``-H`` argument."
  (:require [overtone.core :as o]
            [neurodynamics.synths :as s]
            [neurodynamics.patterns :as p])
  (:gen-class))


(defn- parse-args [argv]
  (let [args (into {} (partition-all 2) argv)]
    {:pattern (or (get args "--pattern") "single")
     :synth   (or (get args "--synth") "mono-saw")
     :hold    (or (some-> (get args "--hold") Integer/parseInt) 0)}))


(defn -main [& argv]
  (let [{:keys [pattern synth hold]} (parse-args argv)
        pattern-data (p/load-pattern pattern)
        synth-def    (s/pick-synth synth)]
    (println "Booting SuperCollider server…")
    (if-let [dev (System/getenv "SC_HW_DEVICE_NAME")]
      (do (println (str "  forcing output device: " dev))
          (o/boot-server :hw-device-name dev))
      (o/boot-server))
    (Thread/sleep 400)  ; let server finish starting
    (try
      (println (format "  playing %s via %s (%.1fs pattern + %d ms tail)"
                       pattern synth
                       (double (:duration_s pattern-data))
                       (+ 800 hold)))
      (p/play-pattern synth-def pattern-data)
      (Thread/sleep (+ (p/total-duration-ms pattern-data) hold))
      (finally
        (o/kill-server)))
    (System/exit 0)))
