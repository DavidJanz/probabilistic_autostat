(ns settings)

(def x-range 50)
(def y-range 20)
(def x* (vec (map vector (range -5 5 0.1))))

(def choices-list [['ADD 2]['MUL 1]['REP 1]['REM 1]['NOACT 3]])

(def proposal-list [['SE 1] ['LIN 1] ['RQ 1] ['PER 1] ['WN 0]])

(def rules [['(+ s s) 2]['(* s s) 2]['b 8]])
(def terms [['WN 0]['LIN 1]['RQ 1]['SE 1]['PER 1]])

(def small-pos-double 1.0E-5)
(def very-neg-double -1.0E10)
