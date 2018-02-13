(ns autostat-mk2.hmc-wrapper
  (:require [clojure.string :as str]
            [clojure.core.matrix :as m]
            )
 (:use [anglican core runtime emit hmc]
       [autostat-mk2 generative-model aux]
       [gp main kernels]
       [settings]
       [clojure.core.matrix.stats]))

(defn wrap-gp-fn [expr struct param-prior flags move-noise gp-var x y]
    (fn [q] 
      (let [params (transform q flags exp-safe)]
        (if (valid? params)
          (m/negate (log-posterior expr struct params param-prior move-noise gp-var x y))                 
          (throw (Exception. 
                  (str "@wrap-gp-fn: NaN or Inf hyper")))))))
 
(defn wrap-gp-grad-fn [expr struct param-prior flags move-noise gp-var x y]
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
                                  prior-grads) 
                log-grads (mapv (fn [g p flag] 
                                 (if (= flag 1) (* p g) g)) 
                               grads params flags)]
            (m/negate log-grads)) ;sorted
           (throw (Exception. 
                   (str "@wrap-gp-grad-fn: NaN or Inf hyper")))))))

(defn move-hypers-hmc [expr hypers gp-var x y markov-steps & 
                       {:keys [debug eps move-noise num-leapfrog-steps] 
                        :or {debug false eps 0.01 move-noise true num-leapfrog-steps 5}}]
  (try 
    (let [;_ (println :move-noise move-noise)
          ordered-hypers (into (sorted-map) hypers)
          struct (h-struct ordered-hypers) ;Struct has NO GP
          template (if move-noise 
                     (merge {'AGP 'WN} (create-dict expr))
                     (create-dict expr))     ; unordered
          flags (if move-noise 
                  (cons 1 (transform-flags ordered-hypers template))
                  (transform-flags ordered-hypers template))    ;Ordered
          param-prior (hyper-prior template 1) ;unordered
          
          start-q (if move-noise 
                    (transform (cons gp-var (h-flatten ordered-hypers)) 
                               flags log-safe)
                    (transform (h-flatten ordered-hypers ) flags log-safe))


          gp-fn (wrap-gp-fn expr struct param-prior flags move-noise gp-var x y)
          gp-grad-fn (wrap-gp-grad-fn expr struct param-prior flags move-noise gp-var x y)
          eps-vec (repeat (count start-q) eps)
          start-grad-q (gp-grad-fn start-q)
          abs-start-grad-q (m/abs start-grad-q)
          eps-div  (m/div eps-vec abs-start-grad-q)
          eps-scaled (mapv #(Math/min (double %1) (double %2)) eps-div eps-vec)
          ;_ (println :abs-start-grad-q abs-start-grad-q)
          ;_ (println :eps-scaled eps-scaled)

          end-q (loop [q start-q 
                       i markov-steps]
                  (if (> i 0)
                    (recur (hmc-transition 
                            gp-fn 
                            gp-grad-fn 
                            eps num-leapfrog-steps 
                            q)
                           (dec i))
                    q))
                  
          params (transform end-q flags exp-safe)]
      [(if move-noise (first params) gp-var) (h-unflatten (if move-noise (rest params) params) struct)])
    (catch Exception e 
      (binding [*out* *err*] (println (str "@move-hypers-hmc: " (.getMessage e)))) 
      [gp-var hypers])))

(defn move-mh [expr hypers gp-var move-noise x y]
  (let [gp (gp-train (make expr hypers) gp-var x y)
        posterior-before (log-posterior gp)
        ordered-hypers (into (sorted-map) hypers)
        struct (h-struct ordered-hypers) ;Struct has NO GP
        template (if move-noise 
                   (merge {'AGP 'WN} (create-dict expr))
                   (create-dict expr))     ; unordered
        flags (if move-noise 
                (cons 1 (transform-flags ordered-hypers template))
                (transform-flags ordered-hypers template))    ;Ordered
        param-prior (hyper-prior template 1) ;unordered
        
        params-start (if move-noise 
                       (cons gp-var (h-flatten ordered-hypers)) 
                       (h-flatten ordered-hypers ))

        start-q (transform params-start flags log-safe)

        proposal (repeatedly (count start-q) #(sample (normal 0 1)))
        
        end-q (m/add start-q proposal)

        params (transform end-q flags exp-safe)

        gp-var (if move-noise (first params) gp-var) 
        hypers (h-unflatten (if move-noise (rest params) params) struct)

        gp (gp-train (make expr hypers) gp-var x y)
        posterior-after (log-posterior gp)
        params-delta (m/sub params params-start)
        prob-forward (reduce + (map #(observe (normal 0 1) %) params-delta))
        prob-backward (reduce + (map #(observe (normal 0 1) %) (m/negate params-delta)))

        accept-move (Math/exp (- prob-forward prob-backward))

        _ (println accept-move)
        log-accept (- (+ prob-forward posterior-after)
                        (+ prob-backward posterior-before))
        accept (Math/exp log-accept)
        _ (println accept)
]
))

(defn move-test-mh [q]
  (let [log-q (Math/log q)
        noise (sample (normal 0 1))
        log-q-after (+ log-q noise)
        q-after (Math/exp log-q-after)
        diff (- q-after q)
        forward (observe (normal 0 1) (Math/exp noise))
        backward (observe (normal 0 1) (Math/exp (- noise)))
        ratio (Math/exp (- forward backward))]
    (* diff ratio)))
