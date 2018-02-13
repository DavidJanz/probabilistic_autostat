(ns anglican.pmc
  "Population Monte Carlo"
  (require [anglican.inference :refer [exec infer checkpoint]]
           [anglican.state :refer [initial-state add-log-weight]]
           [anglican.smc :refer [resample]]
           [anglican.runtime :refer [sample observe]]
           [clojure.core.matrix.stats :refer [mean]]))

(derive ::algorithm :anglican.inference/algorithm)

(defmethod checkpoint [::algorithm anglican.trap.sample] [_ smp]
  (let [s (sample (:dist smp))
        l (- (observe (:dist smp) s))]
    #((:cont smp) s (add-log-weight (:state smp) l))))

(defn retrieve [state k]
  (get-in state [:anglican.state/store k]))
 
(defmethod infer :pmc [_ prog value
                       & {:keys [number-of-particles]  
                          :or {number-of-particles 1}}]
  (assert (>= number-of-particles 1)
          ":number-of-particles must be at least 1")
  (letfn [(sample-seq [values]
            (lazy-seq
              (let [results (doall (pmap (fn [value]
                                   (exec ::algorithm
                                          prog value initial-state))
                                 values))
                    particles (resample results 
                                          (count results))
                    log-weights (map (fn [r] 
                                       (:log-weight (:state r))) 
                                     particles)
                    values (map (fn [r] 
                                  (retrieve (:state r) :value))
                                particles)]
                (print (str (mean log-weights) ","))
                (concat (map :state results) 
                        (sample-seq values)))))]
    (sample-seq (repeat number-of-particles value))))
