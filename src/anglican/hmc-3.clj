(ns anglican.hmc
  "Basic HMC sampler implementation (used in Bayesian optimization)"
  (:require [clojure.core.matrix :as mat
             :refer [matrix mul add sub]])
  (:use anglican.runtime))

(defn- sum 
  [x & [dim & _]]
  (if (and dim (> dim 0))
    (reduce mat/add (mat/slice-views x dim))
    (reduce mat/add x)))

(defn- mean
  [x & [dim & _]]
  (let [dim (or dim 0)]
    (mat/div (sum x dim) (get (mat/shape x) dim)))) 

(defn- sq
  [x]
  (mat/mul x x))

(defn hmc-integrate
  "Preforms leap-frog integration of trajectory."
  [grad-u eps num-steps q p]
  (loop [q q
         p (mat/sub p (mat/mul 0.5 eps (grad-u q)))
         n 1]
    (if (< n num-steps)
      (let [q-new (mat/add q (mat/mul eps p))
            p-new (mat/sub p (mat/mul eps (grad-u q-new)))]
        (recur q-new p-new (inc n)))
      [q (mat/sub p (mat/mul 0.5 eps (grad-u q)))])))

(defn hmc-transition
  "Performs one Hamiltonian Monte Carlo transition update.

  Accepts functions u and grad-u with arguments [q], a parameter eps
  that specifies the integration step size, and a parameter num-steps
  that specifies the number of integration steps.
  
  Returns a new sample q."
  [u grad-u eps num-steps q-start]
  (let [p-start (mat/matrix
                 (map sample 
                      (repeat (count q-start)
                              (normal 0 1))))
        [q-end p-end] (hmc-integrate grad-u eps num-steps 
                                     q-start p-start)
        k-start (* 0.5 (sum (sq p-start)))
        k-end (* 0.5 (sum (sq p-end)))
        accept-prob (exp (+ (- (u q-start) (u q-end)) 
                            (- k-start k-end)))]
    (if (> accept-prob (rand))
      (do (println "accepted: " accept-prob) q-end)
      (do (println "rejected: " accept-prob) q-start))))

(defn hmc-chain
  "Performs Hamiltonian Monte Carlo to construct a Markov Chain   
 
  Accepts functions u and grad-u with arguments [q], a parameter eps
  that specifies the integration step size, and a parameter num-steps
  that specifies the number of integration steps.
  
  Returns a lazy sequence of samples q."
  [u grad-u eps num-steps q-start]
  (let [q-next (hmc-transition u grad-u eps num-steps q-start)]
    (lazy-seq
     (cons q-next (hmc-chain grad-u eps num-steps q-next)))))

;; test code
#_(let [S (matrix [[1 0.1] [0.1 1]])
        u (fn [x] (* 0.5 (mat/mmul (mat/mmul x Sinv) x)))
        grad-u (fn [x] (mat/mmul Sinv x))
        chain (hmc-chain grad-u 0.01 100 [0 0])
        samples (take 1000 chain)]
    ;; this should converge to S
    (mean (map mat/outer-product samples samples)))
