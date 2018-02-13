(ns autostat-mk2.mutations
 (:use [anglican core runtime emit]
       [autostat-mk2.generative-model]
       [settings]))

(defn append-base [expression symbol extension]
  (cond 
    ; Leave symbols as they are
    (symbol? expression) expression
    ; If the expression matches the one we need to expand,
    ; concatenate the extension to it
    (= (second expression) symbol) (concat extension (list expression))
    ; If the expression is a sum/product, map replace-base over it
    ; and simplify the expression as per the generative-model
    (sumprod? (first expression)) (let [sign (first expression)]
                                    (cons sign 
                                          (simplify 
                                           sign 
                                           (map 
                                            #(append-base % symbol extension) 
                                            (rest expression)))))
    ; Else map replace-base over the list
    :else (map #(append-base % symbol extension) expression)))

(defn cond-to-rem [expression symbol]
  (if (and 
       (seq? expression) 
       (= (second expression) symbol)) 
    true false))

(defn strip-extras [result]
  (if (and (sumprod? (first result)) 
           (= (count result) 2)) 
    (second result)
    result))

(defn remove-base [expression symbol] 
  (cond
    ; Leave symbols as they are
    (symbol? expression) expression
    :else (strip-extras (map 
                         #(remove-base % symbol) 
                         (remove 
                          #(cond-to-rem % symbol) 
                          expression)))))            
 
(defn change-base [expression old-symbol new-kernel]
  (cond 
    (symbol? expression) expression
    (= (second expression) old-symbol) new-kernel
    :else (map #(change-base % old-symbol new-kernel) expression)))

(defn rem-from-dict [expr-to-rem dict-vec]
  (remove #(= % expr-to-rem) dict-vec))

(defn rem-from-choices [symbol choices]
  (vec (remove #(= (first %) 'REM) choices)))

(with-primitive-procedures 
  [remove-base append-base remove rem-from-dict rem-from-choices hyper-prior
   change-base]
  (defm mutate-rem [kernel dict]
    (let [; Convert dictionary to a vector
          dict-vec (map reverse (into [] dict))
          ; Pick an item of the dict-vect uniformly
          n-to-extend (sample (discrete (repeat (count dict) 1)))
          expr-to-rem (nth dict-vec n-to-extend)
          sym-to-rem (second expr-to-rem)
          kernel-out (remove-base kernel sym-to-rem)]
      [kernel-out (apply merge (map 
                                #(apply hash-map (reverse %)) 
                                (rem-from-dict expr-to-rem dict-vec)))]))

  (defm mutate-change [kernel dict proposal dims]
    (let [dict-vec (map reverse (into [] dict))
          ; Pick an item of the dict-vect uniformly
          n-to-change (sample (discrete (repeat (count dict) 1)))
          expr-to-change (nth dict-vec n-to-change)
          ; Get this item's symbol
          sym-to-change (second expr-to-change)
          ; Sample a new kernel from proposal
          change-type (sample proposal)
          change-sym (gensym change-type)
          ; Create a kernel from sample, format (+/* (KTYPE KSYMBOL))
          ; We will concatenate this to the chosen item
          new-kernel (list change-type change-sym)
          ; Sample some hyperparameters for the new kernel
          h-prior (hyper-prior {(symbol change-sym) 
                                (symbol change-type)} 
                               dims)
          new-hypers (sample h-prior)
          remmed-list (apply merge (map #(apply hash-map (reverse %)) 
                                      (rem-from-dict expr-to-change dict-vec)))
          new-dict (into (sorted-map) (merge new-hypers remmed-list))]
      [(change-base kernel sym-to-change new-kernel) new-dict]))

  (defm mutate-expand [kernel dict proposal dims sign]
    (let [; Convert dictionary to a vector
          dict-vec (map reverse (into [] dict))
          ; Pick an item of the dict-vect uniformly
          n-to-extend (sample (discrete (repeat (count dict) 1)))
          expr-to-extend (nth dict-vec n-to-extend)
          ; Get this item's symbol
          sym-to-extend (second expr-to-extend)
          ; Sample a new kernel from proposal
          extend-type (sample proposal)
          extend-sym (gensym extend-type)
          ; Create a kernel from sample, format (+/* (KTYPE KSYMBOL))
          ; We will concatenate this to the chosen item
          extend-kernel (list sign (list extend-type extend-sym))
          ; Sample some hyperparameters for the new kernel
          h-prior (hyper-prior {(symbol extend-sym) 
                                (symbol extend-type)} 
                               dims)
          extend-hypers (sample h-prior)]
          ; Return a vector of the modified kernel and
          ; the dict with new the hyperparameters added
      [(append-base kernel sym-to-extend extend-kernel) 
       (merge dict extend-hypers)]))

  (defm mutate-kernel [kernel dict dims]
    (let [; If kernel has only 1 base kernel, disallow REM
          choices (if (= 2 (count kernel)) 
                    (rem-from-choices 'REM choices-list)
                    choices-list)
          proposal (categorical proposal-list)]
      (case (sample (categorical choices))
        ADD (mutate-expand kernel dict proposal dims '+) 
        MUL (mutate-expand kernel dict proposal dims '*)
        REP (mutate-change kernel dict proposal dims)
        REM (mutate-rem kernel dict)
        NOACT [kernel dict]))))
