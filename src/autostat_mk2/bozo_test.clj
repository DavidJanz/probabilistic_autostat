(ns autostat-mk2.bozo-test
  (:require [clojure.string :as str]
            [clojure.core.matrix :as m]
            [clojure.core.memoize :as mem])
  (:use [anglican runtime emit hmc]
        [anglican.improper-smc]
        [anglican.core :only [doquery]]
        [autostat-mk2 aux generative-model mutations data hmc-wrapper]
        [gp main kernels]
        [clojure.tools.cli]
        [bozo.core]))
 
(defn wrap-gp-opt [expr struct param-prior flags move-noise gp-var x y]
    (fn [q] 
       (let [params (transform q flags exp-safe)] ;ordered
         (if (valid? params)           
          (let [gp-var (if move-noise (first params) gp-var)
                
                hypers (h-unflatten (if move-noise (rest params) params) struct) ;ordered
                gp (gp-train (make expr hypers) gp-var x y)                
                prior-grads (h-flatten (log-hyper-grads
                                        param-prior
                                        (into (sorted-map) 
                                              (if move-noise (merge {'AGP [gp-var]} hypers) hypers))))
                grads (m/add 
                       (h-flatten (into (sorted-map) 
                                        (if move-noise 
                                          (:log-evidence-grads gp) 
                                          (dissoc (:log-evidence-grads gp) 'AGP)))) 
                                  0 #_prior-grads) ;sorted
                log-grads (mapv (fn [g p flag] 
                                 (if (= flag 1) (* p (- g)) g)) 
                               grads params flags)
                post (:log-evidence gp)
                _ (println :post post)]
             (list (- post) (double-array log-grads)))))))
            

(defn optimize-hypers [expr hypers gp-var x y move-noise]
    (let [ordered-hypers (into (sorted-map) hypers)
          struct (h-struct ordered-hypers)
          template (if move-noise 
                     (merge {'AGP 'WN} (create-dict expr))
                     (create-dict expr))     
          flags (if move-noise 
                  (cons 1 (transform-flags ordered-hypers template))
                  (transform-flags ordered-hypers template))    
          param-prior (hyper-prior template 1) 
          
          start-q (if move-noise 
                    (transform (cons gp-var (h-flatten ordered-hypers)) 
                               flags log-safe)
                    (transform (h-flatten ordered-hypers ) flags log-safe))
          
          start-q (double-array start-q)

          gp-opt-fn (wrap-gp-opt expr struct param-prior flags move-noise gp-var x y)
          
          end-q (lbfgs gp-opt-fn start-q {:eps 0.5})
                  
          params (transform end-q flags exp-safe)]
      [(if move-noise (first params) gp-var) (h-unflatten (if move-noise (rest params) params) struct)]))

(defn opt-one-kernel-test [expr]
  (let [data (load-data "co2.csv")
        x (take 207 (:x data))
        y (take 207 (:y data))
        _ (println :expr expr)
        hypers (sample (hyper-prior (create-dict expr) 1))
        _ (println :hypers-old hypers)
        gp-var 0.2
        
        [gp-var-new hypers-new] (optimize-hypers expr hypers gp-var x y true)]
    (println :hypers-old gp-var hypers)    
    (println :hypers-new gp-var-new hypers-new)
    ;(println :gp-var-new gp-var-new)
    [(make expr hypers-new) gp-var-new]))
