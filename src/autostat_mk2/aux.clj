(ns autostat-mk2.aux
  (:require [clojure.string :as str]
            [clojure.core.matrix :as m])
 (:use [anglican runtime emit]
       [autostat-mk2 generative-model data]
       [gp kernels main]
       [settings]))

(defn exp-safe [x]
  (let [exp-x (Math/exp x)]
    (if (< exp-x small-pos-double)
      small-pos-double
      exp-x)))

(defn log-safe [x]
  (Math/log x))

(defn NaN? [x]
  (false? (== x x)))

(defn round2 [val]
  (double (/ (round (* 100 val)) 100)))

(defn observe* [dist val]
  (observe dist val))

(defn process-query-result [result]
  (let [[expr gp-var hypers]  result
        _ (println :expr expr)
        ;expr (second expr)
        ;hypers (second hypers)
        ;gp-var (second gp-var)
        ]
    [(make expr hypers) gp-var])) 

(defn display-kernel-gp [processed-result src]
  (let [data (load-data src)
        [kernel gp-var] processed-result
        gp (gp-train kernel gp-var (first-frac (:x data) 0.5) (first-frac (:y data) 0.5))
        pred (gp-predict gp (data :x))
        pred-out {:x-s ((:x-rev data) (:x-s pred)), 
                  :f-s ((:y-rev data) (:f-s pred)), 
                  :var ((:y-rev data) (:var pred))}
        gp-out {:x-obs ((:x-rev data) (:x-obs gp)),
                :y-obs ((:y-rev data) (:y-obs gp))}]
    (println "y-pred" (:f-s pred-out))
    (gp-visualise gp-out pred-out)))

(defdist gp-dist-wrapper [] []
  (sample [this] true)
  (observe [this val] val))

(defn exp-add [arg1 arg2]
  (m/exp (m/add arg1 arg2)))

(defn transform [values flags function]
  (map (fn [p flag] 
         (if (= flag 1) (function p) p)) 
       values flags))

(defn transform-flags [hypers template]
  (let [types (mapv #(get template (first %)) hypers)
        transform-vecs (flatten (mapv #(:transform (get prior %)) types))]
    (vec transform-vecs)))

(defn valid? [values] 
  (and 
   (not-any? NaN? values) 
   (every? finite? values)))

(defn log-posterior 
  ([gp]
   (let [gp-var (:var-n gp)
         [expr hypers] (listify (:kernel gp))
         template (merge {'AGP 'WN} (create-dict expr))
         param-prior (hyper-prior template 1)
         log-prior-hypers (observe 
                           param-prior 
                           (merge {'AGP [gp-var]} hypers))
         log-prior-expr (observe apcfg expr)]
     (double (+ (:log-evidence gp) log-prior-hypers log-prior-expr))))

  ([expr struct params param-prior move-noise gp-var x y]
   (let [gp-var (if move-noise (first params) gp-var)
         hypers (h-unflatten (if move-noise (rest params) params) struct)
         log-prior-hypers (observe 
                           param-prior 
                           (into (sorted-map) 
                                 (if move-noise (merge {'AGP [gp-var]} hypers) hypers)))
         log-prior-expr (observe apcfg expr)
         gp (gp-train (make expr hypers) gp-var x y)]
     (double (+ (:log-evidence gp) log-prior-hypers log-prior-expr)))))



