(ns pmc-infer
  (:require [clojure.string :as str]
            [anglican.core :refer [doquery]])
 (:use [anglican runtime emit]
       [autostat-mk2 aux data generative-model hmc-wrapper mutations]
       [gp main kernels]
       [settings]))

(with-primitive-procedures
  [make create-dict observe* gp-train hyper-prior pcfg 
   gp-dist-wrapper h-struct h-flatten h-unflatten exp-add
   move-hypers-hmc log-posterior valid?
   log-normal] 

  (defquery pmc-infer [x y gp-var expr hypers]
    (let [;; sample an initial expression
          expr (or expr
                   (sample apcfg))
          ;; sample initial hyperparameters
          hypers (or hypers 
                     (sample (hyper-prior 
                              (create-dict expr) 1)))
          [expr hypers] (mutate-kernel expr hypers 1)
          [gp-var hypers] (move-hypers-hmc expr hypers gp-var x y 6)
          gp (gp-train (make expr hypers) gp-var x y)]
      (if (finite? (:log-evidence gp))
        (doall (observe (gp-dist-wrapper) 
                        (log-posterior gp)))
        (doall (observe (gp-dist-wrapper) 
                        very-neg-double)))
      (predict :gp-var gp-var)
      (predict :expr expr)
      (predict :hypers hypers)
      (store :value [x y gp-var expr hypers]))))



