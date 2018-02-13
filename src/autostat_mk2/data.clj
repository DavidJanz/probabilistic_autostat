(ns autostat-mk2.data
  "Data formatted for use with autostat-mk2"
  (:require [clojure.string :as str]
            [clojure.core.matrix :as m]
            [clojure.data.json :as json]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io])
  (use [anglican.stat]
       [settings]))

(m/set-current-implementation :clatrix)

(defn norm-zero-mean [d]
  (let [d-min (m/emin d)
        d-max (m/emax d)
        d-spread (- d-max d-min)
        d-normed (m/div (m/sub d d-min) d-spread)
        d-normed-mean (/ 
                (m/ereduce + d-normed) 
                (count d))
        d (m/sub d-normed d-normed-mean)
        rev-fn (fn [scaled-d] (m/add (m/mul (m/add scaled-d d-normed-mean) d-spread) d-min))]
  [d rev-fn]))

(defn load-data [src]
  (let [raw (csv/read-csv (-> src io/resource slurp))
        x-parsed (mapv #(vector (Double/parseDouble (first %))) raw)
        y-parsed (mapv #(Double/parseDouble (second %)) raw)
        [x x-rev] (norm-zero-mean x-parsed)
        [y y-rev] (norm-zero-mean y-parsed)
        x (m/mul x x-range)
        y (m/mul y y-range)
        x-rev (fn [d] (x-rev (m/div d x-range)))
        y-rev (fn [d] (y-rev (m/div d y-range)))]
    {:x x :y y :x-rev x-rev :y-rev y-rev :x-raw x-parsed :y-raw y-parsed}))

(defn first-frac [d frac]
  (let [d-len (count d)
        d-to-take (Math/round (double (* frac d-len)))]
    (take d-to-take d)))

