(ns gp.main
  "Contains functions used for Gaussian process
  regression with kernels defined in gp-kernels."
  (:require [clojure.string :as str]
            [clojure.core.matrix.linear :as ml]
            [clojure.core.matrix :as m]
            [clojure.data.json :as json]
            [clojure.java.shell :as shell]
            [clojure.core.async :as a]
            [clojure.core.memoize :as mem])
  (:use [gp.kernels])
  (:gen-class))

(shell/with-sh-dir "/home")

(def chol (mem/lru ml/cholesky :lru/threshold 32))

;; use vectorz vector representation
(m/set-current-implementation :clatrix)

;; Gaussian process implemtations
(defrecord gp-obj [kernel var-n x-obs y-obs L alpha log-evidence log-evidence-grads])

(defn find-grad
  "Lowest level gradient function. Input is always a single dimensional
  derivative. Output is the gradient of log marginal likelihood along that
  dimension."
  [common term]
  ; calculate gradient from single derivative
  (m/mul 0.5 (m/trace (m/mmul common term))))

(defn find-kernel-grads
  "Finds gradients with respect to the hyperparameters of 
  a single base kernel. If hyperparameter has multiple
  dimensions then maps across them."
  [common terms]
  (mapv (fn [term]
         ; if this hyperparameter is single dimensional
         (if (= (m/dimensionality term) 2)
           ; find the gradient with respect to it
           (find-grad common term)
           ; else find the gradient with respect to each of its dimensions
           (mapv #(find-grad common %) term)))
       terms))

(defn find-grads
  "Takes a list of derivatives of base kernels with respect to each
  hyperparameter and calculates the gradient of log marginal likelihood
  with respect to each hyperparameter. Uses equation from page 114 in
  Rasmussen 2006."
  [hyper-derivs alpha L]
  (let [inv-L (m/inverse L)
        ; calculate part common to all gradients
        common (m/sub (m/mmul alpha (m/transpose alpha))
                      (m/mmul (m/transpose inv-L) inv-L))]
    (into (sorted-map) (apply merge (mapv #(hash-map (first %) 
                                  (find-kernel-grads common (second %)))
                       hyper-derivs)))))

(defn gp-train
  "Takes a kernel with hyperparameters, a variance for additive 
  noise and a training dataset of x and y values to train a gp.
  Uses algorithm from page 19 of Rasmussen 2006. Expects x to be
  strictly NxD. D dimension is not optional, even if D=1. y is to
  be strictly N dimensional. Does not support multiple output 
  dimensions. Does not support mean functions."
  [kernel var-n x y]
  (let  [y (m/array y)
         x (m/array x)
         ;var-n (+ var-n 0.1)
         [K hyper-derivs] (eval-kernel (->k-sum [(->k-noise var-n 'AGP) kernel]) x x)
         {:keys [L L*]} (try (chol K)
                             (catch Exception e (do 
                                                  (binding [*out* *err*] 
                                                    (println "@cholesky, exception with hypers: " (second (listify kernel))))
                                                  {:L false :L* false})))]
    (if (not (false? L)) 
      (let [alpha (ml/solve L* (ml/solve L y)) ; alpha = L'\(L\y)
            log-evidence (- (* -0.5 (first (m/mmul (m/transpose y) alpha)))
                                          (m/trace (m/log L))
                                          (* 0.5 (count y) (Math/log (* 2 Math/PI))))
            log-evidence-grads (find-grads hyper-derivs alpha L)]
        (->gp-obj kernel var-n x y L alpha log-evidence log-evidence-grads))
       (->gp-obj kernel var-n x y 0 0 (/ -1. 0.) []))))

(defn gp-predict
  "Takes a trained gp object from gp-train and a set of x values x*
  and returns the unaltered x* values, function value predictions 
  at those values and the variance of those predictions."
  [gp x*]
  (let [x* (m/array x*)
        k* (m/array (first (eval-kernel (:kernel gp) (:x-obs gp) x*)))
        f* (m/mmul (m/transpose (:alpha gp)) (m/transpose k*))
        k** (m/eseq (map (fn [x-elem] (first (eval-kernel (:kernel gp) [[x-elem]] [[x-elem]]))) x*))
        v (map (fn [k] (ml/solve (:L gp) (m/transpose k))) k*)
        vdotv (map (fn [x] (m/dot x x)) v)
        var (m/sub k** vdotv)]
    {:x-s x* :f-s f* :var var}))

(defn gp-write-data
  "Writes GP model data to txt file. CSV format.
  Overwrites existing content."
  [file gp prediction]
  (let [x (str/join ","  (:x-obs gp))
        y (str/join "," (:y-obs gp))
        x* (str/join "," (:x-s prediction))
        f* (str/join "," (:f-s prediction))
        var (str/join "," (:var prediction))]
    (spit file (str/join "\r\n" [x y x* f* var]))))

(defn gp-visualise
  "Displays a matplotlib plot of Gaussian process
  regression data.
  REQUIRES: python, numpy, matplotlib, LaTeX. 
  WARNING: tested only on Linux."
  [gp prediction]
  (let [dir (System/getProperty "user.dir")
        file "gp_output_tmp.txt"
        path (str/join (list dir "/tmp/" file))]
    (println :dir dir)
    (a/go
    (gp-write-data path gp prediction)
    (shell/sh
     ; exec gp-visualise.py with python
     "python" "gp-visualise.py"
     ; pass data file path as argument
     path
     ; pass kernel string as argument
     "kernel"
     "hypers"
     ; execute command in the source directory
     :dir (str/join (list dir "/src/autostat_mk2/"))))))
