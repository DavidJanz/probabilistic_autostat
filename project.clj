(defproject autostat_mk2 "0.8-MEM"
  :description "A more Automted Statistician - 4YP Dave Janz"
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/core.async "0.2.374"]
                 [anglican "0.9.0"]
                 [net.mikera/core.matrix "0.51.0"]
                 [net.mikera/core.matrix.stats "0.7.0"]
                 [org.clojure/data.csv "0.1.3"]
                 [clatrix "0.5.0"]
                 [org.clojure/core.memoize "0.5.8"]
                 [bozo/bozo "0.1.1"]]
  :main autostat-mk2.core
  :resource-paths ["queries" "data"]
  :target-path "target/%s"
  :jvm-opts ["-Xmx1200m" "-XX:+UseParallelGC" "-XX:+UseParallelOldGC"]
  :profiles {:uberjar {:aot :all}})

