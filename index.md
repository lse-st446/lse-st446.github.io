![LSE](lse-logo.jpg) 
# ST446 Distributed Computing for Big Data 

### Lent Term 2018

### Instructors

* [Milan Vojnovic](mailto:m.vojnovic@lse.ac.uk), Department of Statistics.  *Office hours*: By appointment, COL 5.05

### Teaching Assistant
* [Christine Yuen](mailto:L.T.Yuen@lse.ac.uk), Department of Statistics.  *Office hours*: Monday 13:00 - 14:00, COL 5.03 (from week 2)

### Course Information

* Lectures on Mondays 10:00–12:00 in TW2.2.04
* Classes on Thursdays 12:30–14:00 in TW2.4.01

No lectures or classes will take place during School Reading Week 6.

| **Week** | **Topic**                            |
|----------|--------------------------------------|
| 1        | [Introduction](#week-1-introduction-to-basic-concepts-and-system-architectures) |
| 2        | [Distributed file systems and key-value stores](#week-2-databases-and-data-storage-systems)                  |
| 3        | [Distributed computation models](#week-3-querying-unstructured-datasets)    |
| 4        | [Structured data management systems](#week-4-querying-structured-datasets)       |
| 5        | [Graph data processing](#week-5-graph-data-processing)                  |
| 6        | _Reading Week_                       |
| 7        | [Stream data processing](#week-7-stream-data-processing) <br> Guest lecturer: Eno Thereska, Principal Engineer, Amazon |
| 8        | [Scalable machine learning I](#week-8-scalable-machine-learning-i) |
| 9        | [Scalable machine learning II](#week-9-scalable-machine-learning-ii) <br> Guest lecturer: Ryota Tomioka, Researcher, Microsoft Research |
| 10       | [AI applications](#week-10-numerical-computations-using-data-flow-graphs) <br> Guest lecturer: Marc Cohen, Software Engineer, Google |
| 11       | [Numerical computations using data flow graphs](#week-11-deployment-of-computation-jobs-in-production)           |




### Course Description

This course will cover the principles of distributed systems for storing and processing big data. This will include the principles of storage systems, databases and data models that are in common use by on-premise data analytics platforms and cloud computing services. The course will cover the principles of computing over large datasets in distributed computing systems involving multi-core processors and cluster computing systems. Students will learn how to perform canonical distributed computing tasks in batch, streaming and graph processing computation models and how to run scalable machine learning algorithms for regression, classification, clustering and collaborative filtering tasks. This course uses a project-based learning approach where students gain hands-on experience with writing and running computer programmes through computer workshop exercises and project assignments. This will equip students with key skills and knowledge about modern computation platforms for processing big data. In particular, students will get hands-on experience with using Apache Spark, the fastest-growing general engine for processing big data that is used across different industries, and connecting Spark programmes with various databases and other systems. The students will work on weekly exercises and project assignments by using revision-control and group collaboration tools such as GitHub. Each student will develop code for solving one or more computation tasks on an input dataset, and will use GitHub for accessing and submitting course materials and assignments.

On the theory side, we will introduce principles of distributed databases, their design objectives, querying paradigms by using MapReduce style of computations, general numerical computations using dataflow graphs, and querying using SQL application programming interfaces. We will consider graph processing algorithms, for querying graph properties and iterative computations using input graph data. We will also introduce the principles of stream processing, how to perform computations and execute queries over a sliding-window of input data stream elements. We will study the principles of scalable machine learning algorithms that are based on parallel implementations of gradient descent style algorithms for minimizing a loss function, used for training regression and classification models. We will also consider distributed MapReduce based computations for training clustering models such as k-means and collaborative filtering models based on matrix factorization. We will consider numerical computations using dataflow graphs, with a focus on the use case of learning a deep neural network for image classification and other classification tasks. Students will be encouraged to work with computations and data relevant to their own interests.

On the practical side, we will cover a variety of tools that are part of a modern data scientist's toolkit, including distributed computing using Apache Spark, Mapreduce style processing of big data sets, application programming interfaces for querying structured and unstructured datasets, stream data processing, and deploying large-scale machine learning models. You will learn how to write programmes to define Spark jobs using the Python API and how to deploy a Spark job in a production environment. You will learn how to connect Spark data structures with a variety of external data sources, including key-value databases, relational databases, and publish-subscribe messaging systems.

For the final project, we will ask you to develop and run a distributed computation for a given dataset, which you will be expected to implement in a PySpark Jupyter notebook.



### Organization

This course is an introduction to the fundamental concepts of distributed computing for big data for students and assumes no prior knowledge of these concepts.  

The course will involve 20 hours of lectures and 15 hours of computer workshops in the LT. 	


### Prerequisites

Some basic prior programming experience is expected. Prior experience with Python programming is desirable; for example, acquired through the compulsory courses of the MSc in Data Science program.


### Software

We will use some tools, notably Apache Spark general engine for computing over large distributed datasets, PySpark (Python API for Spark), SQL APIs for querying datasets, and Tensorflow library for dataflow programmes. Lectures and assignments will be posted on Github, Students are expected to use Github also to submit problem sets and final exam.

Where appropriate, we will use Jupyter notebooks for lab assignments, demonstrations, and the course notes themselves.

### Assessment

Project assignment (80%) and continuous assessment in weeks 4 and 7 (10% each). Students will be expected to produce 10 problem sets in the LT. 


### Schedule

---
#### Week 1. Introduction

In the first week, we will provide an overview of basic concepts starting with a definition of big data, followed by an overview of structured, semi-structured and unstructured data types, including relational database tables, delimiter-separated file formats such as csv and tsv files, JSON and XML file formats. We will then consider main properties of traditional relational database management systems, their support of transactions and ACID properties. This will lead us to consider the need for the design of scalable systems, the concepts of horizontal and vertical scaling, and various computer system bottlenecks. We will then go on to consider modern big data analytics systems and how they have evolved over time. We will introduce various computation paradigms such as batch processing, interactive processing, stream processing, and lambda architectures. We will discuss main developments such as mapreduce computation model and noSQL databases. The rest of the lecture is focused on discussion of various computational tasks that led to the development of modern big data analytics systems, which will be studied throughout the course. 


*Readings*:
* Hamilton, J., [One Size Doesn't Fit All](http://perspectives.mvdirona.com/2009/11/one-size-does-not-fit-all/), Blog, 2012


*Lab*: **Getting started** 
* Command line interface and commands
* Cluster and bucket creation in a cloud
* Submitting a "Hello World" job on a cluster
* Running a Jupyter notebook on a cluster
* Use of Docker containers

---
#### Week 2. Distributed file systems and key-value stores

In this week, we will first consider the main design principles of distributed file systems, explaining the original Google File System and its refinements, as well as other distributed file systems such as Hadoop Distributed File System (HDFS). We will then consider the main design principles of distributed key-value stores such as Amazon Dynamo and columnar data storage systems such as BigTable and Apache Cassandra. 

*Readings*:
* White, T., [Hadoop: The Definitive Guide](https://www.amazon.co.uk/Hadoop-Definitive-Guide-Tom-White/dp/1491901632/ref=sr_1_1?ie=UTF8&qid=1514806006&sr=8-1&keywords=hadoop+the+definitive+guide), O'Reilly, 4th Edition, 2015
* Carpenter, J. and Hewitt, E., [Cassandra: The Definitive Guide](https://www.amazon.co.uk/Cassandra-Definitive-Guide-Jeff-Carpenter/dp/1491933666/ref=sr_1_1?ie=UTF8&qid=1514834275&sr=8-1&keywords=cassandra+definitive+guide), 2nd Edition, O'Reilly, 2016 
* Ghemawat, S., Gobioff, H. and Leung S.-T., [The Google file system](https://research.google.com/archive/gfs.html), SOSP 2003
* DeCandia, G., Hastorun, D., Jampani, M., Kakulapati, G., Lakshman, A., Pilchin, A., Sivasubramanian, S., Voshall, P. and Vogels, W., [Dynamo: Amazon’s Highly Available Key-value Store](http://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf), SOSP 2007


*Further Readings*:
* [GFS: Evolution on Fast-Forward](http://queue.acm.org/detail.cfm?id=1594206), ACM Queue, Vol 7, No 7, August, 2009
* Apache Hadoop docs: [HDFS Architecture](http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html) 
* Vogels, W., [Amazon's Dynamo](http://www.allthingsdistributed.com/2007/10/amazons_dynamo.html), Blog, 2007
* Nishtala et al, Scaling Memcache at Facebook, NSDI 2013
* Fitzpatrick, [Distributed Caching with Memcached](http://www.linuxjournal.com/article/7451), 2004
* Chang et al, [Bigtable: A distributed storage system for structured data](https://static.googleusercontent.com/media/research.google.com/en//archive/bigtable-osdi06.pdf), OSDI 2006
* Lakshman, A. and Malik, K., [A Decentralized Structured Storage System](http://www.cs.cornell.edu/projects/ladis2009/papers/lakshman-ladis2009.pdf), LADIS 2009
* Calder, B. et al, [Windows Azure Storage: A Highly Available Cloud Storage Service with Strong Consistency](http://sigops.org/sosp/sosp11/current/2011-Cascais/printable/11-calder.pdf), SOSP 2011
* Huang, C., Simitci, H., Xu, Y., Ogus, A., Caider, B., Gopalan, P., Li, J. and Yekhanin, S., [Erasure Coding in Windows Azure Storage](https://www.usenix.org/node/168894), USENIX 2012

*Lab*: **System installation and API practice**
* Go through Hadoop installation
* Basic file manipulation commands working with HDFS
* Reading and writing data from BigTable

---
#### Week 3. Distributed computation models

In this lecture we will explain the basic principles of distributed computation models that are in common use in the context of big data analytics systems. We will start with explaining mapreduce computation model that is in widespread use for distribuged processing of large datasets. We will then move on to consider Pregel computation model, developed for iterative computations such as computing PageRank of a large-scale input graph. Finally, we will consider the concept of a resilient distributed dataset, distributed memory abstraction that lets programmers perform in-memory computations on large clusters in a fault-tolerant manner. This will involve to consider the types of operations performed on resilient distributed datasets and their execution on a distributed cluster of machines. 

*Readings*:
* Karau, H., Konwinski, A., Wendell, P. and Zaharia, M., Learning Spark: Lightining-fast Data Analysis, O'Reilly, 2015
* Karau, H. and Warren, R., High Performance Spark: Best Practices for Scaling & Optimizing Apache Spark, O'Reilly, 2017
* Drabas, T. and Lee D., Learning PySpark, Packt, 2016
* Dean, J. and Ghemawat, S., [Mapreduce: Simiplified Data Processing on Large Clusters](https://courses.cs.washington.edu/courses/cse547/17sp/content/Downloads/p107-dean.pdf), Communications of the ACM, Vol 51, No 1, January 2008; original [OSDI 2004 paper](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf)
* Zaharia M. et al, [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](http://people.csail.mit.edu/matei/papers/2012/nsdi_spark.pdf), NSDI 2012

*Further Readings*:
* Apache Hadoop [documentation](http://hadoop.apache.org/docs/r3.0.0/)
* Apache Hadoop documentation: [MapReduce Tutorial](http://hadoop.apache.org/docs/r3.0.0/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html)
* Malewicz G. et al, [Pregel: A System for Large-Scale Graph Processing](https://www.cs.cmu.edu/~pavlo/courses/fall2013/static/papers/p135-malewicz.pdf), SIGMOD 2010
* Spark programming [guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html)   
* Chambers B. and Zaharia M., Spark: The Definitive Guide, databricks, 2017
* Spark documentation: [PySpark package](http://spark.apache.org/docs/2.1.0/api/python/pyspark.html)

*Lab*: **Mapreduce and resilient distributed datasets**
* Run a mapreduce job on Hadoop for word count on dblp data
* Hands-on experience with running operations on resilient distributed datasets in PySpark, such as map, flatMap, filter, distinct, sample, leftOuter and repartition, and actions such as take, collect, reduce, count, saveAsTextFile and foreach
* Run the word count example using resilient distributed datasets in PySpark

---
#### Week 4. Structured data management systems

In this week we will consider systems for big data analytics using structured query languages. We will start with considering the main architectural principles of Apache Hive, a data warehouse solution running on top of Apache Hadoop. We will consider data types and query language used by Hive. We will then consider the main design principles that underlie Dremel (BigQuery) and Spark SQL for querying data using structured query languages. 

*Readings*:
* White, T., [Hadoop: The Definitive Guide](https://www.amazon.co.uk/Hadoop-Definitive-Guide-Tom-White/dp/1491901632/ref=sr_1_1?ie=UTF8&qid=1514806006&sr=8-1&keywords=hadoop+the+definitive+guide), O'Reilly, 4th Edition, 2015
* Karau, H., Konwinski, A., Wendell, P. and Zaharia, M., [Learning Spark: Lightining-fast Data Analysis](https://www.amazon.co.uk/Learning-Spark-Lightning-Fast-Data-Analysis/dp/1449358624), Chapter 9 Spark SQL, O'Reilly, 2015
* Karau, H. and Warren, R., [High Performance Spark: Best Practices for Scaling & Optimizing Apache Spark](https://www.amazon.co.uk/High-Performance-Spark-Practices-Optimizing/dp/1491943203/ref=pd_lpo_sbs_14_t_1?_encoding=UTF8&psc=1&refRID=6QE38GNZN2YBJ5SS99FP), O'Reilly, 2017
* Drabas, T. and Lee D., [Learning PySpark](https://www.amazon.co.uk/Learning-PySpark-Tomasz-Drabas/dp/1786463709), Chapter 3 DataFrames, Packt, 2016
* Rutherglen, J., Wampler, D., Capriolo, E., [Programming Hive](https://www.safaribooksonline.com/library/view/programming-hive/9781449326944/), 2nd Edition, O'Reilly, 2017
* Prokopp, C., [The Free Hive Book](https://github.com/Prokopp/the-free-hive-book)

*Further Readings*:
* Apache Hive [Tutorial](https://cwiki.apache.org/confluence/display/Hive/Tutorial)
* Apache Hive [Language Manual](https://cwiki.apache.org/confluence/display/Hive/LanguageManual)
* Thusoo, A., Sarma, J.-S., Jain, N., Shao, Z., Chakka, P., Zhang, N., Antony S., Liu, H.
and Murthy R., [Hive-A Petabyte Scale Data Warhouse Using Hadoop](http://infolab.stanford.edu/~ragho/hive-icde2010.pdf), ICDE 2010
* Spark SQL programming guide: [Spark SQL, DataFrames and Datasets Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html), Apache Spark 2.2.0, 2017
* Armbrust, M., et al, [Spark SQL: Relational Data Processing in Spark](https://amplab.cs.berkeley.edu/wp-content/uploads/2015/03/SparkSQLSigmod2015.pdf), ACM SIGMOD 2015
* Armbrust, M., [Dataframes: Simple and Fast Analysis of Structured Data](http://go.databricks.com/databricks-webinar-spark-dataframes-simple-and-fast-analysis-of-structured-data-0), Webinar, 2017
* Big Query SQL Reference: [Standard](https://cloud.google.com/bigquery/docs/reference/standard-sql/) and [Legacy](https://cloud.google.com/bigquery/docs/reference/legacy-sql)

*Lab*: **Hive and Spark SQL queries**
* Run Hive queries, basic standard SQL and Hive specific queries such as TRANSFORM, and MAP and REDUCE, queries
* Loading data from sources, including JSON, XML, weblogs using regular expressions 
* Running queries in Spark SQL using dataframe API and Spark Session sql API
* Data management using BigQuery via web interface and connector with Python

Note: assignment for grading to be given in this week

---
#### Week 5. Graph data processing

In this week we will consider principles and systems for scalable processing of large-scale graph data. This include queries such as evaluating node centralities (e.g. degree centrality), graph traversal or motif queries for finding structural in graph data (e.g. identifying friends-of-friends of a person who were born in London), and iterative algorithms on graph input data (e.g. computing PageRank node centralities). We will discuss different data models for representation of graph data such as [RDF](http://www.w3.org/TR/rdf-sparql-query/), as well as query languages, including [SPARQL](http://www.w3.org/TR/rdf-sparql-query/), [Gremlin](http://tinkerpop.apache.org/), [Cypher](https://neo4j.com/developer/cypher/) and [openCyper](http://www.opencypher.org/) that used in graph databases. We will introduce the bulk synchronous parallel computation model that underlies the design of modern computation platforms for iterative computing on input graph data. 


*Readings*:
* Drabas, T. and Karau, H., _Learning PySpark_, Chapter 7: GraphFrames, Packt, 2017
* Spark [GraphX: programming guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
(http://giraph.apache.org/)
* Gonzalez, J. E. et al, [GraphX: Graph Processing in Distributed Dataflow Framework](https://www.usenix.org/node/186217), OSDI 2014

*Further Resources*:

* Malewicz, G. et al, [Pregel: A System for Large-Scale Graph Processing](https://kowshik.github.io/JPregel/pregel_paper.pdf), ACM SIGMOD 2010; open source cousin: [Apache Giraph]
* Low, Y. et al, [Distributed GraphLab: A Framework for Machine Learning and Data Mining in the Cloud](http://vldb.org/pvldb/vol5/p716_yuchenglow_vldb2012.pdf), VLDB 2012
* Valiant, L. G., [A Bridging Model for Parallel Computation](http://web.mit.edu/6.976/www/handout/valiant2.pdf), Communications of the ACM, Vol 3, No 8, August 1990.

*Lab*: **Analysis of StackExchange user-posts-topics relations**
* Importing StackExchange relations into graphframes
* Computing degree and PageRank node centralities
* Answering graph motif queries
* Breadth-first search and connected components


---
#### Week 6. Reading week

---
#### Week 7. Stream data processing

In this week we will consider the basic concepts of data stream processing systems. We will explain various global aggregators, cumulative, and sliding-window stream data processing tasks. We will introduce the concept of publish-subscribe systems and use Apache Kafka as an example. We will discuss the importance of fault tolerance in stream processing systems and discuss fault tolerance models such as execute exactly-once. In this context, we will discuss the guarantees provided by Zookeper, an open source server which enables highly reliable distributed coordination.  

*Readings*:
* Spark [streaming programming guide](http://spark.apache.org/docs/latest/streaming-programming-guide.html)
* Spark [structured streaming programming guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)


*Further Resources*:
* Zaharia, M. et al, [Discretized Streams: Fault-Tolerant Streaming Computation at Scale](http://people.csail.mit.edu/matei/papers/2013/sosp_spark_streaming.pdf), SOSP 2013
* Apache [Kafka documentation](https://kafka.apache.org/documentation/)

*Lab*: **Twitter feed processing**
* Import Twitter feed into Kafka as a topic
* Integrate Spark with Kafka
* Track heavy-hitter topics
* Visualize topic trends

---
#### Week 8. Scalable machine learning I

In this week we will introduce the basic concepts of distributed machine learning algorithms for regression and classification tasks. We will discuss batch optimization methods for model parameter estimation by using gradient descent methods and its variations such as [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) and [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS). We will also cover online optimisation methods such as stochastic gradient descent (SGD), parallel SGD and mini-batch SGD methods. We will discuss as model and data paralellisation models. 

*Readings*:
* Bottou, L. and Le Cun, Y., [Large Scale Online Learning](http://papers.nips.cc/paper/2365-large-scale-online-learning), NIPS 2003
* Drabas, T. and Lee, D.  _Learning PySpark_, Chapter 5: Intoducing MLib, Packt, 2017
* Spark programming guide [MLib: RDD-based API](https://spark.apache.org/docs/latest/mllib-guide.html) 

*Further Resources*:
* Li, M., [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf), OSDI 2014
* [Numerical optimization: understanding L-BFGS](http://aria42.com/blog/2014/12/understanding-lbfgs), blog, December 2, 2014
* Machine Learning [Glossary](https://developers.google.com/machine-learning/glossary)

*Lab*: **Spark ML package and churn prediction task**
* Basic features of Spark MLib package 
* Churn prediction:
   * Import the Orange Telecoms Churn dataset
   * Train a logistic regression model, compute predictions
   * Train a decision tree model, compute predictions
   * Evaluate and compare the two models

Note: assignment for grading to be given in this week

---
#### Week 9. Scalable machine learning II

In this week we will continue by considering distributed machine learning algorithms for clustering and collaborative filtering tasks. We will discuss a Mapreduce algorithm for k-means clustering problem, as well as an iterative algorithm for a collaborative filtering problem. We will consider Spark API approaches provided by MLib and ML packages. For the latter, we will introduce the concept of a pipeline that consists of a dataflow passing through transformer and estimator operators. 


*Readings*:
* Murphy, K. P., _Machine Learning: A Probabilistic Perspective_, k-means, Section 11.4.2.5, The MIT Press, 2012  
* Drabas, T. and Lee, D.  _Learning PySpark_, Chapter 5: Intoducing MLib and Chapter 6: Introducting the ML Package, Packt, 2017
* Spark programming guide [MLib: RDD-based API](https://spark.apache.org/docs/latest/mllib-guide.html) 

*Further Resources*:
* Moore, A., [K-means and Hierarchical Clustering Tutorial](https://www.autonlab.org/tutorials/kmeans.html)
* Wang, Q., [Spark machine learning pipeline by example](https://community.hortonworks.com/articles/53903/spark-machine-learning-pipeline-by-example.html), August 31, 2016
* Zadeh, R. et al, [Matrix Computations and Optimizations in Spark](http://www.kdd.org/kdd2016/subtopic/view/matrix-computations-and-optimization-in-apache-spark), KDD 2016

*Lab*: **Clustering and movies recommendation**
* k-means clustering using a Mapreduce algorithm
* Movie recommendations using MovieLens data and training a collaborative filtering model using Alternating Least Square (ALS) algorithm

---
#### Week 10. AI applications

Guest lecture: "Democratizing AI," Marc Cohen, Software Engineer, Google. 

The lecture will provide an overview of cloud services provided by Google including TensorFlow, vision API, translation API, video intelligence API, cloud ML engine, and managed TensorFlow at scale.

*Lab*: **Using APIs for solving AI tasks**
* TBD

---
#### Week 11. Numerical computations using dataflow graphs

In our last lecture, we will introduce the basic concepts of performing numerical computations using data flow graphs. In such settings, the graph nodes represent mathematical operations, while the graph edges represent the multidimensional data arrays that flow between them. We will explain the architecture of Tensorflow, an open source library for numerical computations using data flow graphs. We will go over the the use case of learning a deep neural network, taking the basic architecture of a feedforward deep neural network.

*Readings*:
* TensorFlow [API docs](https://www.tensorflow.org/api_docs/)
* Abadi et al, [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi), OSDI 2016
* Drabas, T. and Lee, D., _Learning PySpark_, Chapter 8: TensorFrames, Packt, 2017
* TensorFrames [gitHub page](https://github.com/databricks/tensorframes)

*Further Resources*:
* Tensorflow [gitHub page](https://github.com/tensorflow/tensorflow)
* Abadi et al, [A computational model for TensorFlow: an introduction](https://dl.acm.org/citation.cfm?doid=3088525.3088527), MAPL 2017
* Dean, J., [Large-Scale Deep Learning with Tensorflow](https://www.matroid.com/scaledml/slides/jeff.pdf), ScaledML 2016
* Yu, D. et al, [An Introduction to Computational Networks and the Computational Network Toolkit](https://www.microsoft.com/en-us/research/publication/an-introduction-to-computational-networks-and-the-computational-network-toolkit/), Microsoft Research Technical Report, 2014

*Lab*: **Distributed Tensorflow**
* Import a dataset of labeled images
* Specify a feedforward deep neural network model
* Train the model
* Evaluate the classification accuracy of the trained model
