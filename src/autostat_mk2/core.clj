(ns autostat-mk2.core
  (:require [clojure.string :as str]
            [clojure.core.matrix :as m]
            [clojure.core.memoize :as mem])
  (:use [anglican runtime emit hmc]
        [anglican.pmc]
        [anglican.core :only [doquery]]
        [autostat-mk2 aux generative-model mutations data hmc-wrapper bozo-test]
        [gp main kernels]
        [clojure.tools.cli])
  (:gen-class))

(defn process-results [result data frac]
  (let [[gp-var expr hypers] (:predicts result)
        expr (second expr)
        hypers (second hypers)
        gp-var (second gp-var)
        kernel (make expr hypers)
        gp (gp-train kernel gp-var  (first-frac (:x data) frac) (first-frac (:y data) frac))
        pred (gp-predict gp (:x data))
        pred-y-scaled ((:y-rev data) (:f-s pred))
        pred-v-scaled ((:y-rev data) (:var pred))]
    (locking *out* 
      (do (println (str "posterior|" (pr-str expr) "," (pr-str gp-var) "," (pr-str hypers)))
          (println (str "y-prediction|" (str/join "," pred-y-scaled)))
          (println (str "v-prediction|" (str/join "," pred-v-scaled)))
          0))))

#_(defn test-pmc-infer [number-of-particles]
  (let [data (load-data "airline.csv")
        [x y] (map data [:x :y])
        samples (doquery :pmc 
                         pmc-infer
                         [x y 0.1 nil nil]
                         :number-of-particles number-of-particles
                         :warmup false)]
    (nth samples (* 2 number-of-particles))))

(defn -main [& args]
  (require 'pmc-infer)
  (let [queries (find-ns 'pmc-infer)
        [opts args banner] (cli args 
                                ["-p" "--particles" "Number of particles"
                                 :default 8 :flag false 
                                 :parse-fn #(int (Integer. %))]
                                ["-n" "--noise" "GP noise value"
                                 :default 0.15 :flag false 
                                 :parse-fn #(double (Double. %))]
                                ["-f" "--fraction" "Data fraction"
                                 :default 0.2 :flag false 
                                 :parse-fn #(double (Double. %))]
                                ["-d" "--data" "Dataset"
                                 :default "co2.csv" :flag false]
                                ["-b" "--betas" "Number of betas"
                                 :default 2 :flag false
                                 :parse-fn #(int (Integer. %))])
        betas (:betas opts)
        data (load-data (:data opts))
        x (first-frac (:x data) (:fraction opts))
        y (first-frac (:y data) (:fraction opts))
        particles (:particles opts)]

    (println (str "particles|" (:particles opts)))
    (println (str "data|" (:data opts)))
    (println (str "data-fraction|" (:fraction opts)))
    (println (str "data-count|" (Math/round (* (:fraction opts) 
                                               (count (:y data))))))  
    (println (str "betas|" (:betas opts)))
    (println (str "noise|" (:noise opts)))
    (print "pred-probabilities|")

    (let [samples (doquery :pmc 
                         (ns-resolve queries 'pmc-infer)
                         [x y (:noise opts) nil nil]
                         :number-of-particles particles
                         :warmup false)
          result (drop (* particles (dec betas)) (take (* particles betas) samples))]
      (doall result)
      (println)
      (flush)
      (println (pr-str result))
      (println (str "xdata|" (str/join "," (m/eseq (:x-raw data)))))
      (println (str "ydata|" (str/join "," (vec (:y-raw data)))))
      (let [status (reduce + (pmap #(process-results % data (:fraction opts)) result))]
        (println (str "exit|" status) )))))




