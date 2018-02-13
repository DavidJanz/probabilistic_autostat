(ns autostat-mk2.generative-model
  "Generative grammar for sampling kernels and
  hyperparameters. External interface uniform with
  those of other Anglican distributions. Note, where
  probability is outputted or referenced in comments, 
  the quantity is actually the log probability."
  (:require [clojure.string :as str]
            [clojure.core.matrix :as m])
  (:use [anglican.runtime]
        [settings]))

(def catalan-numbers [1 1 2 5 14 42 132 429 1430 4862 16796 58786])

(defn- sample-log-normal [this]
  (let [z (sample (normal 0 1))]
    (exp (+ (:location this) (* (:scale this) z)))))

(defn- observe-log-normal [this value]
    (observe (normal (:location this) (:scale this)) (log value)))

(defn log-log-normal-grad [dist value]
  (let [value (if (== value 0) small-pos-double value)]
    (/ 
     (- 
      (:location dist)
      (pow (:scale dist) 2) 
      (log value))
     (* (pow (:scale dist) 2) value)))) 

(defn log-normal-grad [dist value]
  (/ (- (:mean dist) value) (pow (:sd dist) 2)))

(defdist log-normal [location scale] []
  (sample [this] (sample-log-normal this))
  (observe [this value] (observe-log-normal this value)))

(def prior {'SE {:hypers [(log-normal 0 1.5) 
                          [(log-normal 1 1.5)]]
                 :grads [#(log-log-normal-grad (log-normal 0 1.5) %)
                         [#(log-log-normal-grad (log-normal 1 1.5) %)]]
                 :transform [1 [1]]}
            'WN {:hypers [(log-normal 0 1.5)]
                 :grads [#(log-log-normal-grad (log-normal 0 1.5) %)]
                 :transform [1]}
            'RQ {:hypers [(log-normal 0 1.5) 
                          [(log-normal 1 1.5)] 
                          (log-normal 1 2)]
                 :grads [#(log-log-normal-grad (log-normal 0 1.5) %)
                         [#(log-log-normal-grad (log-normal 1 1.5) %)]
                         #(log-log-normal-grad (log-normal 1 2) %)]
                 :transform [1 1 1]}
            'LIN {:hypers [(log-normal 0 1.5) 
                           (log-normal 0 2) 
                           [(normal 0 3)]]
                  :grads [#(log-log-normal-grad (log-normal 0 1.5) %)
                          #(log-log-normal-grad (log-normal 0 2) %)
                          [#(log-normal-grad (normal 0 5) %)]]
                  :transform [1 1 [0]]}
            'PER {:hypers [(log-normal 0 1.5) 
                           [(log-normal 0 1.5)] 
                           [(log-normal -0.5 2)]]
                  :grads [#(log-log-normal-grad (log-normal 0 1.5) %)
                          [#(log-log-normal-grad (log-normal 0 1.5) %)]
                          [#(log-log-normal-grad (log-normal -0.5 2) %)]]
                  :transform [1 [1] [1]]}})

#_(def prior {'SE {:hypers [(log-normal 0 1) 
                          [(log-normal 0 1)]]
                 :grads [#(log-log-normal-grad (log-normal 0 1) %)
                         [#(log-log-normal-grad (log-normal 0 1) %)]]
                 :transform [1 [1]]}
            'WN {:hypers [(log-normal 0 1)]
                 :grads [#(log-log-normal-grad (log-normal 0 1) %)]
                 :transform [1]}
            'RQ {:hypers [(log-normal 0 1) 
                          [(log-normal 0 1)] 
                          (log-normal 0 1)]
                 :grads [#(log-log-normal-grad (log-normal 0 2) %)
                         [#(log-log-normal-grad (log-normal 2 2) %)]
                         #(log-log-normal-grad (log-normal 0 0.8) %)]
                 :transform [1 1 1]}
            'LIN {:hypers [(log-normal 0 1) 
                           (log-normal 0 1) 
                           [(normal 0 3)]]
                  :grads [#(log-log-normal-grad (log-normal 0 1) %)
                          #(log-log-normal-grad (log-normal 0 1) %)
                          [#(log-normal-grad (normal 0 3) %)]]
                  :transform [1 1 [0]]}
            'PER {:hypers [(log-normal 0 1) 
                           [(log-normal 0 1)] 
                           [(log-normal 0 1)]]
                  :grads [#(log-log-normal-grad (log-normal 0 1) %)
                          [#(log-log-normal-grad (log-normal 0 1) %)]
                          [#(log-log-normal-grad (log-normal 0 1) %)]]
                  :transform [1 [1] [1]]}})

(defn sumprod? 
  "Checks if input is an add or mul symbol."
  [expr-type]
  (or (= expr-type '+) (= expr-type '*)))

(defn create-dict [expression]
  (let [expr-type (symbol (first expression))]
    (if
        ; if we are dealing with a sum/product, recursively
        ; evaluate each subexpression and merge their maps
        (sumprod? expr-type) (apply merge (map create-dict (rest expression)))
        ; otherwise we have a base kernel. Create a map with 
        ; its symbol and type.
        {(second expression) expr-type})))

(defn simplify [symbol expression] 
  (loop [expression expression
         result (list)]
    (if (empty? expression)
      ; if we're finished, return result
      result
      ; otherwise take first element of the expression
      (let [cur (first expression)]
        (cond 
          ; if it is a symbol, prepend it to result and carry on
          ; note: cannot be merged into :else as if cur a symbol
          ; the next condition, if evaluated, will throw an exception.
          (symbol? cur) (recur 
                         (rest expression) 
                         (cons cur result))
          ; if it is the same sign as that of the outer expression
          ; simplify it by extracting its terms and appending them 
          ; to the result
          (= (first cur) symbol) (recur 
                                  (rest expression) 
                                  (concat (rest cur) result))
          ; otherwise we cannot simplify the expression, so prepend
          ; and carry on
          :else (recur 
                 (rest expression) 
                 (cons cur result)))))))

(defn- sample-pcfg   
  [this expression]
  (cond 
    ; if we are at a terminal, return list of prefix and symbol
    (= expression 'b) (let [term (sample (:terms-dist this))] 
                        (list term (gensym term)))
    ; if we are at a non-terminal, use a rule to replace it
    ; and then evaluate the replacement
    (= expression 's) (sample-pcfg this (sample (:rules-dist this)))
    ; otherwise we are at a sum/product list
    :else (cons 
           ; keep the add/mul sign
           (first expression) 
           (->> 
            ; sample-pcfg for each subexpression
            ; that the product/sum is over
            (map 
             #(sample-pcfg this %) 
             (rest expression))
            ; then if any of the subexpressions have the same
            ; sign as the expression, merge them into the expression
            (simplify (first expression))))))

(defn- observe-pcfg [this expression]
  (let [expr-type (symbol (first expression))]
    (if (sumprod? expr-type)
      ; If expression is a sum or product
      (+ ; return the sum of 
       (* ; the probability of drawing a sum/product of this many terms
        (observe (:rules-dist this)  (list (first expression) 's 's))
        (dec (count (rest expression))))
          ; and the probability of each individual term (recursive)
       (get catalan-numbers (count (rest expression)))
       (reduce + (map #(observe this %) (rest expression))))
      ; Otherwise the is a base kernel term
      (+ ; so sum the probabilities of each term being terminal
       (observe (:rules-dist this) 'b) 
       ; and of it in turn being a specific base kernel.
       (observe  (:terms-dist this) expr-type)))))

(defdist pcfg
  "A pcfg with a hardcoded non-terminal, s. Terminals argument
  takes a vector of [terminal, weight] tuples. Rules takes a vector
  of [replacement value, weight] tuples. Sample/observe functions 
  externalised for ease of recursion."
  [rules terminals] [rules-dist (categorical rules)
                     terms-dist (categorical terminals)]
  (sample [this] (sample-pcfg this 's))
  (observe [this expression] (observe-pcfg this expression)))

(def apcfg (pcfg rules terms))

(defn- sample-hypers [expr dims]
  (let [prior-vec (:hypers (get prior (second expr)))
        hypers (mapv (fn [dist] (if (vector? dist) 
                                  (vec (repeatedly 
                                        dims 
                                        #(sample (first dist))))
                                  (sample dist))) prior-vec)]
    {(first expr) hypers}))

(defn log-hyper-grads-base [dict-entry template]
  (let [prior-type (get template (apply first dict-entry))
        grad-fn-vec (:grads (get prior prior-type))
        hypers (apply second dict-entry)
        grads (mapv (fn [fn val] (if (vector? fn)
                                       (mapv #((first fn) %) val)
                                       (fn val))) grad-fn-vec hypers)]
  {(apply first dict-entry) grads}))

(defn log-hyper-grads [prior dict]
  (let [template (:template prior)
        grads (map #(log-hyper-grads-base 
                     (apply hash-map %) 
                     template) dict)]
    (into (sorted-map) (apply merge grads))))

(defn- observe-hypers [dict-entry template]
  (let [prior-type (get template (first dict-entry))
        prior-vec (:hypers (get prior prior-type))
        hypers (second dict-entry)
        observes (mapv (fn [dist val] (if (vector? dist)
                                       (mapv #(observe (first dist) %) val)
                                       (observe dist val))) prior-vec hypers)]
  (reduce + (flatten observes))))

(defn transform-hypers [dict-entry template transform-fn]
  (let [prior-type (get template (first dict-entry))
        transform-vec (:transform (get prior prior-type))
        hypers (second dict-entry)
        transformed (mapv (fn [trans val] (if (vector? trans)
                                            (if (= 1 (first trans)) 
                                              (mapv transform-fn val) val)
                                            (if (= 1 trans) 
                                              (transform-fn val) val))) 
                          transform-vec hypers)]
    {(first dict-entry) transformed}))

(defdist hyper-prior
  "Prior distribution over hyperparameters in a given kernel structure.
  Requires all hyperparameters to take the same base prior."  
  [template dims] []
  (sample [this] (into (sorted-map) (apply merge (map 
                                                  #(sample-hypers % dims) 
                                                  template))))
  (observe [this dict] (reduce + (map #(observe-hypers % template) 
                                        dict))))

(defn entry-struct [dict-entry] 
  {(first dict-entry) (mapv #(if (number? %) 0 (count %)) (second dict-entry))})

(defn h-struct 
  "Outputs the hyperparameter structure of a kernel. Encodes scalars with 0.
  Vectors noted by their count. Used for non-destructive flattening."
  [hypers] 
  (let [structs (apply merge (mapv entry-struct hypers))]
    (into (sorted-map) structs)))

(defn h-flatten 
  "Flattens a dictionary of hyperparameters."
  [hypers]
  (let [hypers-sorted (into (sorted-map) hypers)]
    (vec (flatten (mapv second hypers-sorted)))))

(defn- h-unflatten-cur 
  "Takes a vector of hyperparameters and a length count for each
  hyperparameter in current base kernel. Returns a formatted 
  hyperparameter vector and vector of hyperparameters minus
  the values that were used."
  [hypers len]
  ; Loop over each kernel hyperparameter type
  (loop [hypers hypers
         len len
         cur-result []]
    ; If there isn't another hyperparameter to process
    (if (empty? len)
      ; return
      [cur-result hypers]
      ; otherwise
      (if (= (first len) 0)
        ; if it is a scalar, take one from hypers,
        ; conj it to the result and recur
        (let [c (first hypers) hypers (next hypers)] 
          (recur hypers (next len) (conj cur-result c)))
        ; else take len from hypers, put them in a vector, conj and recur
        (let [c (vec (take (first len) hypers)) hypers (nthnext hypers (first len))] 
          (recur hypers (next len) (conj cur-result c)))))))

(defn h-unflatten 
  "Restores hyperparameter dict from a structure and flattened vector." 
  [hypers-vec struct]
  ; Loop over each structure element
  (loop [hypers hypers-vec
         struct struct
         result {}]
    ; Run h-unflatten-cur to process the current structure element
    ; and merge its output into the result hashmap
    (let [cur (first struct)
          [cur-result hypers-rest] (h-unflatten-cur hypers (second cur)) 
          result (merge result {(first cur) cur-result})]
          (if (empty? (rest struct)) 
            result 
            (recur hypers-rest (rest struct) result)))))

