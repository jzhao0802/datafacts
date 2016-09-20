name := "SAMPLE"

version := "1.0"

scalaVersion := "2.10.4"

resolvers += "Akka Repository" at "http://repo.akka.io/releases/"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.5.2" % "provided",
  "org.apache.spark" % "spark-mllib_2.10" % "1.5.2" % "provided"
)