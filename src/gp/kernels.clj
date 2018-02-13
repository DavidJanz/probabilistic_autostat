(ns gp.kernels
  "Contains kernel template, implementations
  and helper functions, for use with gp-main
  Gaussian process regression implementation.
  Also contains functionality for converting
  from string to kernels and vice versa."
  (:require [clojure.string :as str]
            [clojure.core.matrix :as m]
            [clojure.core.memoize :as mem])
  (:use [anglican emit])
  (:gen-class))

(m/set-current-implementation :clatrix)

;; Kernel calculations helper functions
(defn in? 
  "true if coll contains elm"
  [coll elm]  
  (some #(= elm %) coll))

(defn bcast
  "Takes inputs that are N1xD and N2xD and 
  broadcasts func across them such that the 
  result is N1xN2xD."
  [func x1 x2]
  (let [[N1 D] (m/shape x1)
        [N2 _] (m/shape x2)
        x1 (m/slice-views (m/broadcast x1 [N2 N1 D]) 2)
        x2 (map m/transpose (m/slice-views (m/broadcast x2 [N1 N2 D]) 2))]
    (map #(func (m/array %1) (m/array %2)) x1 x2)))
  
(def bcast (mem/lru bcast :lru/threshold 32))

(defn scaled-dist
  "Takes an R^NxNxD set of distances across each
  individual dimension (x1_d - x2_d) and len_d for
  that dimension. Divides each dimension distance 
  by the relevant lenthscale and sums across the 
  dimensions. Output is a scaled distance in R^NxN."
  [x-diff scale]
    (apply m/add (map #(m/div %1 %2) x-diff scale)))

(def scaled-dist (mem/lru scaled-dist :lru/threshold 32))

;; Kernel definitions and implementations
(defprotocol Kernel
  (eval-kernel [this x1 x2])
  (eval-grad [this x1 x2 cache])) 

(defrecord k-se [var-n len id])
(extend-protocol Kernel k-se
  (eval-grad [this x1 x2 cache]
    (let [var-deriv (:k-exp cache) 
          len-deriv-factor (map #(m/div %1 %2) (:sq-x-diff cache) (m/pow (:len this) 3))
          len-deriv (mapv #(m/mul % (:k cache)) len-deriv-factor)]
      {(:id this) [var-deriv len-deriv]}))
  (eval-kernel [this x1 x2]
    (let [x-diff (bcast m/sub x1 x2)
          sq-x-diff (map #(m/pow % 2) x-diff)
          sq-len (m/pow (:len this) 2)
          exponent (m/negate 
                        (m/div (scaled-dist 
                                sq-x-diff 
                                sq-len)
                               2))
          k-exp (m/exp exponent)
          k (m/mul k-exp (:var-n this))]
      [k  (eval-grad this x1 x2 {:k k :k-exp k-exp :sq-x-diff sq-x-diff})])))

(defrecord k-rq [var-n len alpha id])
(extend-protocol Kernel k-rq
  (eval-grad [this x1 x2 cache]
    (let [var-deriv (:Z-pow cache)
          len-Z (m/pow (:Z cache) (+ (- (:alpha this)) (- 1)))
          len-deriv-factor (map #(m/mul (:var-n this) 
                                        (m/div %1 (m/pow %2 3)))
                                (:r-sq cache) (:len this))
          len-deriv (mapv #(m/mul % len-Z) len-deriv-factor)
          alpha-deriv-factor (m/mul (:var-n this) (m/sub (m/div (m/sub (:Z cache) 1) (:Z cache)) (m/log (:Z cache))))
          alpha-deriv (m/mul (m/pow (:Z cache) (- (:alpha this))) alpha-deriv-factor)]
      {(:id this) [var-deriv len-deriv alpha-deriv]}))
  (eval-kernel [this x1 x2]
    (let [r (bcast m/sub x1 x2)
          r-sq (map #(m/pow % 2) r)
          dist (scaled-dist r-sq (m/mul 2 (:alpha this) (m/pow (:len this) 2)))
          Z (m/add 1 dist)
          Z-pow (m/pow Z (- (:alpha this)))
          k (m/mul Z-pow (:var-n this))]
      [k  (eval-grad this x1 x2 {:Z Z :r-sq r-sq :Z-pow Z-pow})])))

(defrecord k-per [var-n len per id])
(extend-protocol Kernel k-per
  (eval-grad [this x1 x2 cache]
    (let [var-deriv (:k-exp cache)
          
          len-deriv-factor (map #(m/div (m/mul 4 (m/pow (m/sin %1) 2)) %2) (:angle cache) (m/pow (:len this) 3)) 
          per-deriv-factor (map #(m/div (m/mul 2 Math/PI %1 (m/sin (m/mul 2 %2))) 
                                        (m/mul %3 %4)) (:r-abs cache) (:angle cache) (m/pow (:len this) 2) (m/pow (:per this) 2))
          len-deriv (mapv #(m/mul % (:k cache)) len-deriv-factor)
          per-deriv (mapv #(m/mul % (:k cache)) per-deriv-factor)]
      {(:id this) [var-deriv len-deriv per-deriv]}))
  (eval-kernel [this x1 x2] 
    (let [r-abs (map m/abs (bcast m/sub x1 x2))
          angle (map #(m/mul Math/PI (m/div %1 %2)) r-abs (:per this)) 
          
          distance (scaled-dist (map #(m/pow (m/sin %) 2) angle) (m/pow (:len this) 2)) 
          k-exp (m/exp (m/negate (m/mul 2 distance)))
          k (m/mul k-exp (:var-n this))]  
      [k (eval-grad this x1 x2 {:k k :k-exp k-exp :r-abs r-abs :angle angle})])))

(defrecord k-noise [var-n id])
(extend-protocol Kernel k-noise
  (eval-grad [this x1 x2 cache]
    {(:id this) [(:id-m cache)]})
  (eval-kernel [this x1 x2]
    (if (= (m/shape x1) (m/shape x2))
      (let [id-m (m/identity-matrix (first (m/shape x1)))
            k (m/mul (:var-n this) id-m)]
        [k (eval-grad this x1 x2 {:id-m id-m})])
      (let [[x-short x-long] (if (< (first (m/shape x1)) (first (m/shape x2))) [x1 x2] [x2 x1])
            x-var (mapv #(if (in? x-long %) [(:var-n this)] [0]) x-short)
            x-other (mapv #(if (in? x-short %) [1] [0]) x-long)
            k (first (bcast m/mul x-var x-other))]
        [k {(:id this) nil}]))))

(defrecord k-lin [var-n offset intercept id]) 
(extend-protocol Kernel k-lin
  (eval-grad [this x1 x2 cache]
    (let [var-deriv (:k-unscaled cache)
          x-sum (bcast m/add x1 x2)
          intercept-deriv (m/mul 
                           (:var-n this) 
                           (mapv #(m/sub %1 %2) (m/mul 2 (:intercept this)) x-sum))
          offset-deriv (m/broadcast 1 (m/shape x-sum))] 
      {(:id this) [var-deriv offset-deriv intercept-deriv]}))
  (eval-kernel [this x1 x2]
    (let [x1-c (map #(m/sub %1 %2) (m/slice-views x1 1) (:intercept this))
          x2-c (map #(m/sub %1 %2) (m/slice-views x2 1) (:intercept this))
          prod (bcast m/mul (m/transpose x1-c) (m/transpose x2-c))
          k-unscaled (reduce m/add prod)
          k  (m/mul k-unscaled (:var-n this))]
      [(m/add k (:offset this)) (eval-grad this x1 x2 {:k k :k-unscaled k-unscaled})])))
 
(defrecord k-sum [children])
(extend-protocol Kernel k-sum
  (eval-grad [this x1 x2 k])
  (eval-kernel [this x1 x2]
    (let [child-results (map #(eval-kernel % x1 x2) (:children this))
          k (reduce m/add (map first child-results))
          grads (apply merge (map second child-results))]
      [k grads])))

(defrecord k-prod [children])
(extend-protocol Kernel k-prod
  (eval-grad [this x1 x2 k])
  (eval-kernel [this x1 x2]
     (let [child-results (map #(eval-kernel % x1 x2) (:children this))
           k (reduce m/mul (map first child-results))
           grads (apply merge (map second child-results))]
       [k grads])))

;; Strings <-> Kernels conversion
(defn make
  ([expression] (make expression nil))
  ([expression dict]
   (let [[k-expr & sub-exprs] expression
         k-loopup {'SE "se" 
                   'LIN "lin" 
                   'WN "noise" 
                   'PER "per"
                   'RQ "rq"}] 
     (case k-expr 
       + (->k-sum (mapv #(make % dict) sub-exprs))
       * (->k-prod (mapv #(make % dict) sub-exprs))
       (do
         (apply 
          (resolve (symbol (str "gp.kernels/->k-" (get k-loopup k-expr))))
        (if (nil? dict) 
          (concat sub-exprs (list (gensym (str k-expr))))
          (concat (get dict (symbol (first sub-exprs))) sub-exprs))))))))

(defn listify 
([kernel] (listify kernel {}))
([kernel dict]
  
  (cond
    (instance? k-se kernel)    [(list 'SE (:id kernel)) 
                                (merge dict {(:id kernel) 
                                             [(:var-n kernel) (:len kernel)]})]
    (instance? k-rq kernel)    [(list 'RQ (:id kernel)) 
                                (merge dict {(:id kernel) 
                                             [(:var-n kernel) (:len kernel) (:alpha kernel)]})]
    (instance? k-lin kernel)   [(list 'LIN (:id kernel)) 
                                (merge dict {(:id kernel) 
                                             [(:var-n kernel) (:offset kernel) (:intercept kernel)]})]
    (instance? k-noise kernel) [(list 'WN (:id kernel)) 
                                (merge dict {(:id kernel) 
                                             [(:var-n kernel)]})]
    (instance? k-per kernel)   [(list 'PER (:id kernel)) 
                                (merge dict {(:id kernel) 
                                             [(:var-n kernel) 
                                              (:len kernel)
                                              (:per kernel)]})]
    (instance? k-prod kernel)  (let [result (map #(listify % dict) (:children kernel))
                                     result-expr (map first result)
                                     result-hypers (into (sorted-map) (merge (map second result)))] [(cons '* result-expr) result-hypers])
    (instance? k-sum kernel)   (let [result (map #(listify % dict) (:children kernel))
                                     result-expr (map first result)
                                     result-hypers (into (sorted-map) (merge (map second result)))] [(cons '+ result-expr) result-hypers]))))
