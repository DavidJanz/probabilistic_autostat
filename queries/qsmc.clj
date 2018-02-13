(ns qsmc
  (:require [clojure.string :as str])
 (:use [anglican runtime emit]
       [autostat-mk2 aux hmc-wrapper generative-model mutations]
       [gp main kernels]
       [settings]))

(with-primitive-procedures
  [make create-dict observe* gp-train hyper-prior pcfg 
   gp-dist-wrapper h-struct h-flatten h-unflatten exp-add
   move-hypers-hmc log-posterior valid?
   log-normal] 

  (defquery smc-infer [x y gp-var betas]
    (let [; sample an initial expression
          expr (sample apcfg)
          ; sample initial hyperparameters
          hypers (sample (hyper-prior 
                          (create-dict expr) 1))]
      (loop [expr expr
             hypers hypers
             gp-var gp-var
             betas betas]
        (if (empty? betas)
          (do
            (predict :gp-var gp-var)
            (predict :expr expr)
            (predict :hypers hypers))
          (let [[expr hypers] (mutate-kernel expr hypers 1)
                [gp-var hypers] (move-hypers-hmc expr hypers gp-var x y 6)
                
                gp (gp-train (make expr hypers) gp-var x y)]
              (if (finite? (:log-evidence gp))
                (do
                  (doall (observe (gp-dist-wrapper) (* (first betas) (log-posterior gp))))
                  (recur expr hypers gp-var (rest betas)))
                (doall (observe (gp-dist-wrapper) very-neg-double)))))))))


